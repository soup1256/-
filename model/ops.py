import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, 0, 0.01)
            init.constant_(module.bias, 0)

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        return self.shifter(x)

class MergeRun(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(MergeRun, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        return c_out + x

    def _initialize_weights(self):
        init_weights(self.modules())

class MergeRunDual(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(MergeRunDual, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        return c_out + x

    def _initialize_weights(self):
        init_weights(self.modules())

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.body(x)

    def _initialize_weights(self):
        init_weights(self.modules())

class BasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.body(x)

    def _initialize_weights(self):
        init_weights(self.modules())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        out = self.body(x)
        return F.relu(out + x)

    def _initialize_weights(self):
        init_weights(self.modules())

class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )
        self._initialize_weights()

    def forward(self, x):
        out = self.body(x)
        return F.relu(out + x)

    def _initialize_weights(self):
        init_weights(self.modules())

class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                    nn.ReLU(inplace=True),
                    nn.PixelShuffle(2)
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(3)
            ]
        self.body = nn.Sequential(*modules)
        self._initialize_weights()

    def forward(self, x):
        return self.body(x)

    def _initialize_weights(self):
        init_weights(self.modules())
