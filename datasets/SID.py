from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
import glob
import numpy as np

import torch
import random
import cv2


def glob_file_list(root):
	return sorted(glob.glob(os.path.join(root, '*')))


def flip(x, dim):
	indices = [slice(None)] * x.dim()
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
	return x[tuple(indices)]



def augment_torch(img_list, hflip=True, rot=True):
	hflip = hflip and random.random() < 0.5
	vflip = rot and random.random() < 0.5
	def _augment(img):
		if hflip:
			img = flip(img, 2)
		if vflip:
			img = flip(img, 1)
		return img
	return [_augment(img) for img in img_list]


class ImagesDataset2(Dataset):
	def __init__(self, source_root_pre, target_root_pre, opts, target_transform=None, source_transform=None, train=1):

		self.source_transform = source_transform
		self.target_transform = target_transform

		source_root = '/mnt/proj73/xgxu/sid_dataset/short_sid2'
		target_root = '/mnt/proj73/xgxu/sid_dataset/long_sid2'
		source_root = os.path.join(source_root_pre, source_root)
		target_root = os.path.join(target_root_pre, target_root)

		self.source_paths = []
		self.target_paths = []
		subfolders_LQ_origin = glob_file_list(source_root)
		subfolders_GT_origin = glob_file_list(target_root)
		subfolders_LQ = []
		subfolders_GT = []
		if train:
			for mm in range(len(subfolders_LQ_origin)):
				name = os.path.basename(subfolders_LQ_origin[mm])
				if '0' in name[0] or '2' in name[0]:
					subfolders_LQ.append(subfolders_LQ_origin[mm])
					subfolders_GT.append(subfolders_GT_origin[mm])
		else:
			for mm in range(len(subfolders_LQ_origin)):
				name = os.path.basename(subfolders_LQ_origin[mm])
				if '1' in name[0]:
					subfolders_LQ.append(subfolders_LQ_origin[mm])
					subfolders_GT.append(subfolders_GT_origin[mm])

		self.train = train
		for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
			subfolder_name = os.path.basename(subfolder_LQ)

			img_paths_LQ = glob_file_list(subfolder_LQ)
			img_paths_GT_origin = glob_file_list(subfolder_GT)
			length = len(img_paths_LQ)
			img_paths_GT = []
			for mm in range(length):
				img_paths_GT.append(img_paths_GT_origin[0])
			self.source_paths.extend(img_paths_LQ)
			self.target_paths.extend(img_paths_GT)

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = np.load(from_path)
		from_im = from_im[:, :, [2,1,0]]
		from_im = Image.fromarray(from_im)

		to_path = self.target_paths[index]
		to_im = np.load(to_path)

		to_im_gray = cv2.cvtColor(to_im, cv2.COLOR_BGR2GRAY)
		sketch = cv2.GaussianBlur(to_im_gray, (3, 3), 0)

		v = np.median(sketch)
		sigma = 0.33
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		sketch = cv2.Canny(sketch, lower, upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		sketch = cv2.dilate(sketch, kernel)

		sketch = np.expand_dims(sketch, axis=-1)
		sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
		assert len(np.unique(sketch)) == 2

		to_im = to_im[:, :, [2,1,0]]
		to_im = Image.fromarray(to_im)

		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)

		if self.train:
			if random.randint(0, 1):
				to_im = flip(to_im, 2)
				from_im = flip(from_im, 2)
				sketch = cv2.flip(sketch, 1)

		to_im=(to_im+1)*0.5
		from_im=(from_im+1)*0.5
		
		height = to_im.shape[1]
		width = to_im.shape[2]
		sketch[sketch == 255] = 1
		sketch = cv2.resize(sketch, (width, height))
		sketch = torch.from_numpy(sketch).permute(2, 0, 1)
		sketch = sketch[0:1, :, :]
		sketch = sketch.long()
		return from_im, to_im, sketch
