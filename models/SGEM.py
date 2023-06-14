import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np

class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, ks, norm_nc, label_nc):
        super().__init__()

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out



#########################################################################################################
def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(0.1,inplace=True)
    )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock2(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock2(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out

def init_upconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)

def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output

def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook
#########################################################################################################

class GlobalGenerator3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator3, self).__init__()
        activation = nn.ReLU(True)

        n_downsampling = 2
        model1 = [nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1), activation]
        ### downsample
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), activation]
        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), activation]

        ### resnet blocks
        model4 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model4 += [ResnetBlock22(ngf * mult, padding_type=padding_type, activation=activation)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

        ks_2d = 3
        self.ks_2d = ks_2d

        mult = 2 ** (n_downsampling - 0)
        self.deconv1 = nn.Conv2d(ngf * mult * 2, int(ngf * mult / 2)*4, kernel_size=3, stride=1, padding=1)
        self.deconv11 = nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1)

        ch3 = int(ngf * mult / 2)
        ks = 3

        self.conv1_1 = conv(1, ch3, kernel_size=ks, stride=1)
        self.fac_deblur1 = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.norm1 = SPADE('instance', 3, int(ngf * mult / 2), 1)
        ############################################
        mult = 2 ** (n_downsampling - 1)
        self.deconv2 = nn.Conv2d(ngf * mult * 2, int(ngf * mult / 2)*4, kernel_size=3, stride=1, padding=1)
        self.deconv22 = nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1)
        ch3 = int(ngf * mult / 2)
        ks = 3

        self.conv2_1 = conv(1, ch3, kernel_size=ks, stride=1)
        self.fac_deblur2 = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.norm2 = SPADE('instance', 3, int(ngf * mult / 2), 1)
        ############################################
        model_tail = []
        model_tail += [nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1)]
        model_tail += [activation, nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]
        self.model_tail = nn.Sequential(*model_tail)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.reset_params()

    @staticmethod
    def weight_init(m, init_type='kaiming', gain=0.02, scale=0.1):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.weight.data *= scale
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input, sketch):
        feature1 = self.model1(input)
        feature2 = self.model2(feature1)
        feature3 = self.model3(feature2)
        feature4 = self.model4(feature3)
        feature4 = torch.cat([feature4, feature3], dim=1)

        feature = self.deconv1(feature4)
        feature = self.pixel_shuffle(feature)
        feature = nn.ReLU(True)(feature)

        height = feature.shape[2]
        width = feature.shape[3]
        sketch2 = F.interpolate(sketch, size=(height, width))
        sketch_feature = self.conv1_1(sketch2)
        kernel_deblur = self.fac_deblur1(sketch_feature)

        feature_edge = kernel2d_conv(feature, kernel_deblur, self.ks_2d)
        feature_edge = self.norm1(feature_edge, sketch)
        feature_edge = self.deconv11(feature_edge)
        feature_edge = nn.ReLU(True)(feature_edge)
        feature = feature + feature_edge
        feature = torch.cat([feature, feature2], dim=1)
        ###################################
        feature = self.deconv2(feature)
        feature = self.pixel_shuffle(feature)
        feature = nn.ReLU(True)(feature)

        height = feature.shape[2]
        width = feature.shape[3]
        sketch2 = F.interpolate(sketch, size=(height, width))
        sketch_feature = self.conv2_1(sketch2)
        kernel_deblur = self.fac_deblur2(sketch_feature)

        feature_edge = kernel2d_conv(feature, kernel_deblur, self.ks_2d)
        feature_edge = self.norm2(feature_edge, sketch)
        feature_edge = self.deconv22(feature_edge)
        feature_edge = nn.ReLU(True)(feature_edge)
        feature = feature + feature_edge
        feature = torch.cat([feature, feature1], dim=1)

        return self.model_tail(feature)


class ResnetBlock22(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock22, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
