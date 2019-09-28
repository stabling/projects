import torch.nn as nn
import torch
from torch.nn import functional


# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = DownSample(64, 64)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = DownSample(128, 128)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = DownSample(256, 256)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = DownSample(512, 512)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样
        self.up6 = UpSample()  # nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024 + 512, 512)
        self.up7 = UpSample()
        self.conv7 = DoubleConv(512 + 256, 256)
        self.up8 = UpSample()
        self.conv8 = DoubleConv(256 + 128, 128)
        self.up9 = UpSample()
        self.conv9 = DoubleConv(128 + 64, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        # print(c1.shape)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        # print(c2.shape)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        # print(c3.shape)
        p3 = self.pool3(c3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        # print(c4.shape)
        p4 = self.pool4(c4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        # print(c5.shape)
        up_6 = self.up6(c5)
        # print(up_6.shape)
        # print(c4.shape)
        merge6 = torch.cat([up_6, c4], dim=1)
        # print(merge6.shape)
        c6 = self.conv6(merge6)
        # print(c6.shape)
        up_7 = self.up7(c6)
        # print(up_7.shape)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        # print(c7.shape)
        up_8 = self.up8(c7)
        # print(up_8.shape)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        # print(c8.shape)
        up_9 = self.up9(c8)
        # print(up_9.shape)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # print(c9.shape)
        c10 = self.conv10(c9)
        # print(c10.shape)
        out = nn.Sigmoid()(c10)
        return out
