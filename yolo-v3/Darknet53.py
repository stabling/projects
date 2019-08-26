import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualLayer(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.res = nn.Sequential(
            ConvolutionLayer(in_channel, in_channel // 2, 1, 1, 0),
            ConvolutionLayer(in_channel // 2, in_channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.res(x) + x


class DownsampleLayer(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down = nn.Sequential(
            ConvolutionLayer(in_channel, out_channel, 3, 2, 1)
        )

    def forward(self, x):
        return self.down(x)


class ConvolutionalSet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.set = nn.Sequential(
            ConvolutionLayer(in_channel, out_channel, 1, 1, 0),
            ConvolutionLayer(out_channel, in_channel, 3, 1, 1),

            ConvolutionLayer(in_channel, out_channel, 1, 1, 0),
            ConvolutionLayer(out_channel, in_channel, 3, 1, 1),

            ConvolutionLayer(in_channel, out_channel, 1, 1, 0),
        )

    def forward(self, x):
        return self.set(x)


class UpsampleLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_52 = nn.Sequential(
            ConvolutionLayer(3, 32, 3, 1, 1),
            DownsampleLayer(32, 48),
            ResidualLayer(48),
            DownsampleLayer(48, 54),
            ResidualLayer(54),
            DownsampleLayer(54, 64),
            ResidualLayer(64),
        )

        self.conv_26 = nn.Sequential(
            DownsampleLayer(64, 80),
            ResidualLayer(80),
        )

        self.conv_13 = nn.Sequential(
            DownsampleLayer(80, 96),
            ResidualLayer(96),
        )

        self.set_13 = nn.Sequential(
            ConvolutionalSet(96, 102)
        )

        self.detect_13 = nn.Sequential(
            ConvolutionLayer(102, 64, 3, 1, 1),
            nn.Conv2d(64, 60, 1, 1, 0)
        )

        self.up_26 = nn.Sequential(
            ConvolutionLayer(102, 80, 1, 1, 0),
            UpsampleLayer()
        )

        self.set_26 = nn.Sequential(
            ConvolutionalSet(160, 80)
        )

        self.detect_26 = nn.Sequential(
            ConvolutionLayer(80, 64, 3, 1, 1),
            nn.Conv2d(64, 60, 1, 1, 0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionLayer(80, 64, 3, 1, 1),
            UpsampleLayer()
        )

        self.set_52 = nn.Sequential(
            ConvolutionalSet(128, 96)
        )

        self.detect_52 = nn.Sequential(
            ConvolutionLayer(96, 64, 3, 1, 1),
            nn.Conv2d(64, 60, 1, 1, 0)
        )

    def forward(self, x):

        h_52 = self.conv_52(x)
        h_26 = self.conv_26(h_52)
        h_13 = self.conv_13(h_26)

        set_13 = self.set_13(h_13)
        detect_13 = self.detect_13(set_13)

        up_26 = self.up_26(set_13)
        cat_26 = torch.cat((h_26, up_26), dim=1)
        set_26 = self.set_26(cat_26)
        detect_26 = self.detect_26(set_26)

        up_52 = self.up_52(set_26)
        cat_52 = torch.cat((h_52, up_52), dim=1)
        set_52 = self.set_52(cat_52)
        detect_52 = self.detect_52(set_52)

        return detect_13, detect_26, detect_52
