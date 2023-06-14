import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from configs import data_configs

from datasets.LOL_real import ImagesDataset2

from criteria.lpips.lpips import LPIPS1
from models.SGEM import GlobalGenerator3
from models.AM import GlobalGenerator3 as GlobalGenerator1
from models.SAG import Discriminator, FullGenerator
from training.ranger import Ranger

import torch.distributed as dist
import math
import numpy as np
import cv2

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='mean')
    return cost

#######################################################
from skimage.metrics import structural_similarity as ssim_o
from skimage.metrics import peak_signal_noise_ratio as psnr_o
import lpips as lpips_o
def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips_o.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=True):
        if gray_scale:
            score, diff = ssim_o(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim_o(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr_o(imgA, imgB)
        return psnr_val
#######################################################

class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0
        self.device = 'cuda'
        self.rank = torch.distributed.get_rank()
        self.opts.device = self.device
        self.net = FullGenerator(512, 512, 8,
                                 channel_multiplier=2, narrow=1.0, device='cuda').to(self.device)

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS1(net_type=self.opts.lpips_type).to(self.device).eval()

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.refinement = GlobalGenerator1(3, 3, 32, 2, 1).to(self.device) ## appearance modeling
        self.refinement2 = GlobalGenerator3(3+3, 3, 16, 1).to(self.device) ## structure-guided enhancement

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        self.net = nn.parallel.DistributedDataParallel(
            self.net,
            device_ids=[self.opts.local_rank],
            output_device=self.opts.local_rank,
            broadcast_buffers=False,find_unused_parameters=True
        )

        self.refinement = nn.parallel.DistributedDataParallel(
            self.refinement,
            device_ids=[self.opts.local_rank],
            output_device=self.opts.local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

        self.refinement2 = nn.parallel.DistributedDataParallel(
            self.refinement2,
            device_ids=[self.opts.local_rank],
            output_device=self.opts.local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator_style = Discriminator(size=512, channel_multiplier=2,
                                                     narrow=1.0, device='cuda').to(self.device)
            self.discriminator_style_optimizer = torch.optim.Adam(list(self.discriminator_style.parameters()),
                                                                  lr=opts.w_discriminator_lr)

            self.discriminator_style = nn.parallel.DistributedDataParallel(
                self.discriminator_style,
                device_ids=[self.opts.local_rank],
                output_device=self.opts.local_rank,
                broadcast_buffers=False,find_unused_parameters=True
            )

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        dataset_ratio = 1

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           sampler=data_sampler(self.train_dataset, shuffle=True, distributed=True),
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        self.train_dataloader = sample_data(self.train_dataloader)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if self.opts.resume_training_from_ckpt is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)

    def load_from_train_checkpoint(self, ckpt):
        if self.rank == 0:
            print('Loading previous training data...')

        device_id = torch.cuda.current_device()
        ckpt = torch.load(self.opts.resume_training_from_ckpt,
                          map_location=lambda storage, loc: storage.cuda(device_id))

        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.module.load_state_dict(ckpt['state_dict'])
        self.refinement.module.load_state_dict(ckpt['state_dict_refine'])
        self.refinement2.module.load_state_dict(ckpt['state_dict_refine2'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator_style.module.load_state_dict(ckpt['discriminator_style_state_dict'])
            self.discriminator_style_optimizer.load_state_dict(ckpt['discriminator_style_optimizer_state_dict'])
        del ckpt
        torch.cuda.empty_cache()

    def train(self):
        self.net.train()
        self.refinement.train()
        self.refinement2.train()
        epoch = 0

        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}

                if self.is_training_discriminator():
                    loss_dict2 = self.train_discriminator_img(batch)
                    loss_dict = {**loss_dict2}

                x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    o1 = torch.cat([x, y], dim=2)*2-1
                    y_hat_inter_save = y_hat_inter
                    y_hat_inter_save2 = y_hat_sketch.repeat(1, 3, 1, 1)
                    o2 = torch.cat([y_hat_inter_save, y_hat_inter_save2], dim=2)*2-1
                    o3 = torch.cat([y_hat, sketch.repeat(1, 3, 1, 1)], dim=2)*2-1
                    self.parse_and_log_images(id_logs, o1, o2, o3, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
            epoch+=1


    def validate(self):
        self.net.eval()
        self.refinement.eval()
        self.refinement2.eval()
        agg_loss_dict = []
        psnr_all = 0
        psnr_all2 = 0
        psnr_count = 0
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x, y, y_hat_all, latent, y_hat_inter, y_hat_sketch, sketch = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat_all, latent, y_hat_inter, y_hat_sketch, sketch)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            y_hat = y_hat_all
            batch_size = x.shape[0]
            for mm in range(batch_size):
                y_this = y[mm, :, :, :].permute(1, 2, 0)
                y_hat_this = y_hat[mm, :, :, :].permute(1, 2, 0)
                y_hat_inter_this = y_hat_inter[mm, :, :, :].permute(1, 2, 0)

                y_this = torch.clamp(y_this, min=0.0, max=1.0)
                y_hat_this = torch.clamp(y_hat_this, min=0.0, max=1.0)
                y_hat_inter_this = torch.clamp(y_hat_inter_this, min=0.0, max=1.0)

                y_this = (y_this.clone().detach().cpu().numpy())*255.0
                y_hat_this = (y_hat_this.clone().detach().cpu().numpy()) * 255.0
                y_hat_inter_this = (y_hat_inter_this.clone().detach().cpu().numpy()) * 255.0

                y_this = np.clip(y_this, a_min=0.0, a_max=255.0).round().astype(np.uint8)
                y_hat_this = np.clip(y_hat_this, a_min=0.0, a_max=255.0).round().astype(np.uint8)
                y_hat_inter_this = np.clip(y_hat_inter_this, a_min=0.0, a_max=255.0).round().astype(np.uint8)

                psnr_this = calculate_psnr(y_hat_this, y_this)
                psnr_all+=psnr_this
                psnr_this2 = calculate_psnr(y_hat_inter_this, y_this)
                psnr_all2+=psnr_this2
                psnr_count += 1

            o1 = torch.cat([x, y], dim=2)*2-1 ## input and GT
            y_hat_inter_save = y_hat_inter
            y_hat_inter_save2 = y_hat_sketch.repeat(1, 3, 1, 1)
            o2 = torch.cat([y_hat_inter_save, y_hat_inter_save2], dim=2)*2-1 ## I_a and I_S
            o3 = torch.cat([y_hat, sketch.repeat(1, 3, 1, 1)], dim=2)*2-1 ## output I_hat and GT for edge

            self.parse_and_log_images(id_logs, o1, o2, o3,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                self.refinement.train()
                self.refinement2.train()
                return None  # Do not log, inaccurate in first batch

        psnr_average = psnr_all*1.0/psnr_count
        psnr_average2 = psnr_all2*1.0/psnr_count
        print('testing psnr for I_a', psnr_average)
        print('testing psnr for I_hat', psnr_average2)

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        self.refinement.train()
        self.refinement2.train()
        return loss_dict


    ##########
    def test_image_generation(self):
        self.save_dir = 'results_image_generation_LOL_real'
        self.save_dir1 = os.path.join(self.save_dir, 'input')
        self.save_dir2 = os.path.join(self.save_dir, 'gt')
        self.save_dir3 = os.path.join(self.save_dir, 'output')
        if not (os.path.exists(self.save_dir)):
            os.mkdir(self.save_dir)
        if not (os.path.exists(self.save_dir1)):
            os.mkdir(self.save_dir1)
        if not (os.path.exists(self.save_dir2)):
            os.mkdir(self.save_dir2)
        if not (os.path.exists(self.save_dir3)):
            os.mkdir(self.save_dir3)
        self.refinement.eval()
        self.refinement2.eval()
        self.net.eval()
        psnr_all = 0
        psnr_count = 0
        ssim_all = 0
        lpips_all = 0
        count = 0

        measure = Measure(use_gpu=False)
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch = self.forward(batch)

            batch_size = x.shape[0]
            for mm in range(batch_size):
                y_this = y[mm, :, :, :].permute(1, 2, 0)
                x_this = x[mm, :, :, :].permute(1, 2, 0)
                y_hat_this = y_hat[mm, :, :, :].permute(1, 2, 0)

                y_this = (y_this.clone().detach().cpu().numpy()) * 255.0
                x_this = (x_this.clone().detach().cpu().numpy()) * 255.0
                y_hat_this = (y_hat_this.clone().detach().cpu().numpy()) * 255.0
                
                y_this = np.clip(y_this, a_min=0.0, a_max=255.0).astype(np.uint8)
                x_this = np.clip(x_this, a_min=0.0, a_max=255.0).astype(np.uint8)
                y_hat_this = np.clip(y_hat_this, a_min=0.0, a_max=255.0).astype(np.uint8)

                psnr_o, ssim_o, lpips_o = measure.measure(y_hat_this.astype(np.uint8), y_this.astype(np.uint8))
                psnr_this = psnr_o
                psnr_all += psnr_this
                ssim_this = ssim_o
                ssim_all += ssim_this
                lpips_this = lpips_o
                lpips_all += lpips_this
                psnr_count += 1

                cv2.imwrite(os.path.join(self.save_dir1, '%05d.png' % count), x_this[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(self.save_dir2, '%05d.png' % count), y_this[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(self.save_dir3, '%05d.png' % count), y_hat_this[:, :, [2, 1, 0]])
                count += 1
                print(count)

        psnr_average = psnr_all * 1.0 / psnr_count
        print('testing psnr', psnr_average)

        ssim_average = ssim_all * 1.0 / psnr_count
        print('testing ssim', ssim_average)

        lpips_average = lpips_all * 1.0 / psnr_count
        print('testing lpips', lpips_average)


    def test_edge_generation(self):
        self.save_dir = 'results_edge_generation_LOL_real'
        self.save_dir1 = os.path.join(self.save_dir, 'input')
        self.save_dir2 = os.path.join(self.save_dir, 'edge_gt')
        self.save_dir3 = os.path.join(self.save_dir, 'edge_output')
        if not (os.path.exists(self.save_dir)):
            os.mkdir(self.save_dir)
        if not (os.path.exists(self.save_dir1)):
            os.mkdir(self.save_dir1)
        if not (os.path.exists(self.save_dir3)):
            os.mkdir(self.save_dir3)
        self.refinement.eval()
        self.refinement2.eval()
        self.net.eval()
        count = 0

        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch = self.forward(batch)

            batch_size = x.shape[0]
            for mm in range(batch_size):
                x_this = x[mm, :, :, :].permute(1, 2, 0)
                y_hat_this2 = y_hat_sketch[mm, :, :, :].permute(1, 2, 0)

                x_this = (x_this.clone().detach().cpu().numpy()) * 255.0
                y_hat_this2 = (y_hat_this2.clone().detach().cpu().numpy()) * 255.0

                x_this = np.clip(x_this, a_min=0.0, a_max=255.0).astype(np.uint8)
                y_hat_this2 = np.clip(y_hat_this2, a_min=0.0, a_max=255.0).astype(np.uint8)

                cv2.imwrite(os.path.join(self.save_dir1, '%05d.png' % count), x_this[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(self.save_dir3, '%05d.png' % count), y_hat_this2[:, :, :])
                count += 1
                print(count)

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.parameters())
        params += list(self.refinement.parameters())
        params += list(self.refinement2.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

        train_dataset = ImagesDataset2(source_root_pre=dataset_args['train_source_root'],
                                      target_root_pre=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts, train=1)
        test_dataset = ImagesDataset2(source_root_pre=dataset_args['test_source_root'],
                                     target_root_pre=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts, train=0)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.is_training_discriminator():
            fake_pred = self.discriminator_style(y_hat_sketch)
            loss_disc_img = F.softplus(-fake_pred).mean()
            loss_dict['encoder_discriminator_img_loss'] = float(loss_disc_img)
            loss += self.opts.w_discriminator_lambda * loss_disc_img

        if self.opts.l2_lambda > 0:
            loss_l2 = nn.MSELoss()(y_hat, y) * 10
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda

            loss_l22 = nn.MSELoss()(y_hat_inter, y)
            loss_dict['loss_l22'] = float(loss_l22)
            loss += loss_l22 * self.opts.l2_lambda

        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda

            loss_lpips2 = self.lpips_loss(y_hat_inter, y)
            loss_dict['loss_lpips2'] = float(loss_lpips2)
            loss += loss_lpips2 * self.opts.lpips_lambda

        loss_edge = cross_entropy_loss_RCF(y_hat_sketch, sketch, 1.1) * 5.0
        loss_dict['loss_edge'] = float(loss_edge)
        loss += loss_edge
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch):
        x, y, sketch = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        sketch = sketch.to(self.device).float()

        y_hat_inter = self.refinement(x)
        y_hat_inter = y_hat_inter + x
        y_hat_inter = torch.clamp(y_hat_inter, min=0.0, max=1.0)

        x_gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        y_hat_sketch, latent = self.net(x_gray, return_latents=True)

        input_variable = torch.cat([y_hat_inter, x], dim=1)
        y_hat = self.refinement2(input_variable, y_hat_sketch)
        y_hat = y_hat + y_hat_inter
        y_hat = torch.clamp(y_hat, min=0.0, max=1.0)
        return x, y, y_hat, latent, y_hat_inter, y_hat_sketch, sketch

    def forward_no_latent2(self, batch):
        with torch.no_grad():
            x, y, sketch = batch
            x, y = x.to(self.device).float(), y.to(self.device).float()
            sketch = sketch.to(self.device).float()
            x_gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
            y_hat_sketch, latent = self.net(x_gray, return_latents=True)
        return sketch, y_hat_sketch

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            if self.rank == 0:
                self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        if self.rank == 0:
            print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            if self.rank == 0:
                print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)

        if self.rank == 0:
            self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.module.state_dict(),
            'state_dict_refine': self.refinement.module.state_dict(),
            'state_dict_refine2': self.refinement2.module.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_style_state_dict'] = self.discriminator_style.module.state_dict()
                save_dict['discriminator_style_optimizer_state_dict'] = self.discriminator_style_optimizer.state_dict()
        return save_dict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_img_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator_img(self, batch):
        loss_dict = {}
        self.requires_grad(self.discriminator_style, True)

        y, y_hat = self.forward_no_latent2(batch)

        real_pred = self.discriminator_style(y)
        fake_pred = self.discriminator_style(y_hat)
        loss = self.discriminator_img_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_img_loss'] = float(loss)

        self.discriminator_style_optimizer.zero_grad()
        loss.backward()
        self.discriminator_style_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = y.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator_style(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator_style.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0].mean()
            r1_final_loss.backward()
            self.discriminator_style_optimizer.step()
            loss_dict['discriminator_img_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator_style, False)

        return loss_dict


    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            y, y_hat = self.forward_no_latent2(test_batch)

            real_pred = self.discriminator_style(y)
            fake_pred = self.discriminator_style(y_hat)
            loss = self.discriminator_img_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict
