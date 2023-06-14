import math
import random
import functools
import operator
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from models.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, device=self.device)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad, device=self.device)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        self.device = device

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            device='cpu'
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, isconcat=True):
        super().__init__()

        self.isconcat = isconcat
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, channel, height, width).normal_()

        if self.isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            isconcat=True,
            device='cpu'
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            device=device
        )

        self.noise = NoiseInjection(isconcat)
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel * feat_multiplier, device=device)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel, device=device)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False, device=device)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        out = nn.Sigmoid()(out)
        return out


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            isconcat=True,
            narrow=1,
            device='cpu'
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(64 * narrow),
            8: int(64 * narrow),
            16: int(64 * narrow),
            32: int(64 * narrow),
            64: int(32 * channel_multiplier * narrow),
            128: int(16 * channel_multiplier * narrow),
            256: int(8 * channel_multiplier * narrow),
            512: int(4 * channel_multiplier * narrow),
            1024: int(2 * channel_multiplier * narrow),
            2048: int(1 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
        )
        self.to_rgb1 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel * self.feat_multiplier,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    isconcat=isconcat,
                    device=device
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel * self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device
                )
            )

            self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:

            noise = [None] * (2 * (self.log_size - 2) + 1)

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, device=device)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

from models.transformer import BasicUformerLayer, Downsample, InputProj, OutputProj

class Encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, img_size,
                 kernel_size=3, depth=2, num_head=2, win_size=8, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', shift_flag=True,
                 downsample=False, device='cpu'):
        super().__init__()

        self.conv0 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer0 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)

        ####
        self.input_proj0 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj1 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj2 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj3 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj4 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj5 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj6 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj7 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj8 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        self.output_proj0 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj1 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj2 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj3 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj4 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj5 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj6 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj7 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj8 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)

        self.conv1 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer1 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer2 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv3 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer3 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv4 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer4 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv5 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer5 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv6 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer6 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv7 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer7 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        self.conv8 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
        self.transformer8 = BasicUformerLayer(dim=in_channel,
                                              output_dim=in_channel,
                                              input_resolution=(img_size,
                                                                img_size),
                                              depth=depth,
                                              num_heads=num_head,
                                              win_size=win_size,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp,
                                              shift_flag=shift_flag)
        if downsample:
            self.downsample0 = Downsample(in_channel, in_channel)
            self.downsample1 = Downsample(in_channel, in_channel)
            self.downsample2 = Downsample(in_channel, in_channel)
            self.downsample3 = Downsample(in_channel, in_channel)
            self.downsample4 = Downsample(in_channel, in_channel)
            self.downsample5 = Downsample(in_channel, in_channel)
            self.downsample6 = Downsample(in_channel, in_channel)
            self.downsample7 = Downsample(in_channel, in_channel)
            self.downsample8 = Downsample(in_channel, in_channel)
        else:
            self.downsample0 = None
            self.downsample1 = None
            self.downsample2 = None
            self.downsample3 = None
            self.downsample4 = None
            self.downsample5 = None
            self.downsample6 = None
            self.downsample7 = None
            self.downsample8 = None

        self.fusion0 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion1 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion2 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion3 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion4 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion5 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion6 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion7 = ConvLayer(out_channel * 2, out_channel, 1, device=device)
        self.fusion8 = ConvLayer(out_channel * 2, out_channel, 1, device=device)

    def forward(self, input):

        short_range = self.conv0(input)
        input_long = self.input_proj0(input)
        long_range = self.transformer0(input_long)
        if self.downsample0 is not None:
            long_range = self.downsample0(long_range)
        long_range = self.output_proj0(long_range)
        f0 = self.fusion0(torch.cat([short_range, long_range], dim=1))

        height = input.shape[2]
        width = input.shape[3]

        input_center = input[:, :, 1:-1, 1:-1]
        left = input[:, :, 0:-2, 1:-1]
        right = input[:, :, 2:, 1:-1]
        top = input[:, :, 1:-1, 0:-2]
        bottom = input[:, :, 1:-1, 2:]

        left_top = input[:, :, 0:-2, 0:-2]
        right_top = input[:, :, 2:, 0:-2]
        left_bottom = input[:, :, 0:-2, 2:]
        right_bottom = input[:, :, 2:, 2:]


        input1 = F.interpolate(input_center-left, size=(height, width))
        f11 = self.conv1(input1)
        input_long = self.input_proj1(input1)
        f12 = self.transformer1(input_long)
        if self.downsample1 is not None:
            f12 = self.downsample1(f12)
        f12 = self.output_proj1(f12)
        f1 = self.fusion1(torch.cat([f11, f12], dim=1))

        ########
        input2 = F.interpolate(input_center - right, size=(height, width))
        f21 = self.conv2(input2)
        input_long = self.input_proj2(input2)
        f22 = self.transformer2(input_long)
        if self.downsample2 is not None:
            f22 = self.downsample2(f22)
        f22 = self.output_proj2(f22)
        f2 = self.fusion2(torch.cat([f21, f22], dim=1))

        ########
        input3 = F.interpolate(input_center - top, size=(height, width))
        f31 = self.conv3(input3)
        input_long = self.input_proj3(input3)
        f32 = self.transformer3(input_long)
        if self.downsample3 is not None:
            f32 = self.downsample3(f32)
        f32 = self.output_proj3(f32)
        f3 = self.fusion3(torch.cat([f31, f32], dim=1))

        ########
        input4 = F.interpolate(input_center - bottom, size=(height, width))
        f41 = self.conv4(input4)
        input_long = self.input_proj4(input4)
        f42 = self.transformer4(input_long)
        if self.downsample4 is not None:
            f42 = self.downsample4(f42)
        f42 = self.output_proj4(f42)
        f4 = self.fusion4(torch.cat([f41, f42], dim=1))

        ########
        input5 = F.interpolate(input_center - left_top, size=(height, width))
        f51 = self.conv5(input5)
        input_long = self.input_proj5(input5)
        f52 = self.transformer5(input_long)
        if self.downsample5 is not None:
            f52 = self.downsample5(f52)
        f52 = self.output_proj5(f52)
        f5 = self.fusion5(torch.cat([f51, f52], dim=1))

        ########
        input6 = F.interpolate(input_center - right_top, size=(height, width))
        f61 = self.conv6(input6)
        input_long = self.input_proj6(input6)
        f62 = self.transformer6(input_long)
        if self.downsample6 is not None:
            f62 = self.downsample6(f62)
        f62 = self.output_proj6(f62)
        f6 = self.fusion6(torch.cat([f61, f62], dim=1))

        ########
        input7 = F.interpolate(input_center - left_bottom, size=(height, width))
        f71 = self.conv7(input7)
        input_long = self.input_proj7(input7)
        f72 = self.transformer7(input_long)
        if self.downsample7 is not None:
            f72 = self.downsample7(f72)
        f72 = self.output_proj7(f72)
        f7 = self.fusion7(torch.cat([f71, f72], dim=1))

        ########
        input8 = F.interpolate(input_center - right_bottom, size=(height, width))
        f81 = self.conv8(input8)
        input_long = self.input_proj8(input8)
        f82 = self.transformer8(input_long)
        if self.downsample8 is not None:
            f82 = self.downsample8(f82)
        f82 = self.output_proj8(f82)
        f8 = self.fusion8(torch.cat([f81, f82], dim=1))

        f = f1+f2+f3+f4+f5+f6+f7+f8
        out = f0+f
        return out


class FullGenerator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            isconcat=True,
            narrow=1,
            device='cpu'
    ):
        super().__init__()

        channels = {
            4: int(64 * narrow),
            8: int(64 * narrow),
            16: int(64 * narrow),
            32: int(64 * narrow),
            64: int(32 * channel_multiplier * narrow),
            128: int(16 * channel_multiplier * narrow),
            256: int(8 * channel_multiplier * narrow),
            512: int(4 * channel_multiplier * narrow),
            1024: int(2 * channel_multiplier * narrow),
            2048: int(1 * channel_multiplier * narrow)
        }


        self.log_size = int(math.log(size, 2))
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier,
                                   blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow,
                                   device=device)

        conv = [ConvLayer(1, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        self.names2 = ['ecd2%d' % i for i in range(self.log_size - 1)]
        depths = [2, 2, 2, 2, 2, 2, 2]
        num_heads = [1, 2, 4, 4, 4, 4, 4]
        win_sizes = [8, 8, 8, 8, 8, 4, 4]
        count = 0
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device),
                    nn.InstanceNorm2d(out_channel)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            conv2 = [Encoder_block(in_channel, out_channel, img_size=2 ** (i - 1), kernel_size=3,
                                   downsample=True, device=device,
                                   depth=depths[count], num_head=num_heads[count], win_size=win_sizes[count]),
                     nn.InstanceNorm2d(out_channel)]
            setattr(self, self.names2[self.log_size - i + 1], nn.Sequential(*conv2))
            in_channel = out_channel
            count += 1
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))

    def forward(self,
                inputs,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):
        noise = []
        for i in range(self.log_size - 1):
            if i >=1:
                ecd = getattr(self, self.names[i])
                ecd2 = getattr(self, self.names2[i])
                inputs1 = ecd(inputs)
                inputs2 = ecd2(inputs)
                inputs3 = inputs1 + inputs2
                noise.append(inputs3)
                inputs = inputs1
            else:
                ecd = getattr(self, self.names[i])
                inputs = ecd(inputs)
                noise.append(inputs)

        inputs = inputs.view(inputs.shape[0], -1)
        outs = self.final_linear(inputs)
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        noise[-1] = None
        noise[-2] = None
        outs = self.generator([outs], return_latents, inject_index, truncation, truncation_latent, input_is_latent,
                              noise=noise[1:])

        return outs


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1, device='cpu'):
        super().__init__()

        channels = {
            4: int(64 * narrow),
            8: int(64 * narrow),
            16: int(64 * narrow),
            32: int(64 * narrow),
            64: int(32 * channel_multiplier * narrow),
            128: int(16 * channel_multiplier * narrow),
            256: int(8 * channel_multiplier * narrow),
            512: int(4 * channel_multiplier * narrow),
            1024: int(2 * channel_multiplier * narrow),
            2048: int(1 * channel_multiplier * narrow)
        }

        convs = [ConvLayer(1, channels[size], 1, device=device)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, device=device))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out
