import torch
import torch.nn as nn
import functools

class GlobalGenerator3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator3, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1), activation]
        ### downsample
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), activation]
        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        model4 = []
        for i in range(n_blocks):
            model4 += [ResnetBlock22(ngf * mult, padding_type=padding_type, activation=activation)]

        mult = 2 ** (n_downsampling - 0)
        model5 = [nn.Conv2d(ngf * mult*2, int(ngf * mult / 2)*4, kernel_size=3, stride=1, padding=1)]
        self.model5_act = activation
        mult = 2 ** (n_downsampling - 1)
        model6 = [nn.Conv2d(ngf * mult*2, int(ngf * mult / 2) * 4, kernel_size=3, stride=1, padding=1)]
        self.model6_act = activation

        model7 = []
        model7 += [nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1)]
        model7 += [activation, nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
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

    def forward(self, input):
        feature1 = self.model1(input)
        feature2 = self.model2(feature1)
        feature3 = self.model3(feature2)

        feature4 = self.model4(feature3)
        feature4 = torch.cat([feature4, feature3], dim=1)

        feature5 = self.model5(feature4)
        feature5 = self.pixel_shuffle(feature5)
        feature5 = self.model5_act(feature5)
        feature5 = torch.cat([feature5, feature2], dim=1)

        feature6 = self.model6(feature5)
        feature6 = self.pixel_shuffle(feature6)
        feature6 = self.model6_act(feature6)
        feature6 = torch.cat([feature6, feature1], dim=1)

        output = self.model7(feature6)
        return output



# Define a resnet block
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

