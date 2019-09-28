import torch.nn as nn
import torch
import time
import math
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):  # 定义基本的卷积类

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualLayer(nn.Module):  # 定义残差类

    def __init__(self, in_channel):
        super().__init__()
        self.res = nn.Sequential(
            ConvolutionLayer(in_channel, in_channel // 2, 1, 1, 0),
            ConvolutionLayer(in_channel // 2, in_channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.res(x) + x


class DownsampleLayer(nn.Module):  # 定义下采样类

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down = nn.Sequential(
            ConvolutionLayer(in_channel, out_channel, 3, 2, 1)
        )

    def forward(self, x):
        return self.down(x)


class ConvolutionalSet(nn.Module):  # 定义卷积集类

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


class UpsampleLayer(nn.Module):  # 定义上采样类

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class MobileLayer(nn.Module):  # 定义mobilenet类

    def __init__(self, in_size, expand_size, kernel_size, stride):
        super(MobileLayer, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.conv3 = nn.Conv2d(expand_size, in_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_size)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return out


class CombileLayer(nn.Module):
    def __init__(self, in_size, expand_size, kernel_size, stride, n):
        super().__init__()
        self.layer = nn.Sequential(
            *[MobileLayer(in_size, expand_size, kernel_size, stride) for i in range(n)]
        )

    def forward(self, x):
        return x


class Net(nn.Module):  # 定义网络结构

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            # ConvolutionLayer(3, 32, 3, 1),
            # CombileLayer(32, 64, 3, 1, 15),
            # DownsampleLayer(32, 48),
            # ConvolutionLayer(48, 64, 3, 1),
            # CombileLayer(64, 80, 3, 1, 15),
            # DownsampleLayer(64, 128),
            # ConvolutionLayer(128, 160, 3, 1),
            # CombileLayer(160, 256, 3, 1, 15),
            # DownsampleLayer(160, 256),
            # ConvolutionLayer(256, 512, 3, 1)
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(1 * 1 * 512, 128)
        )

        self.cls_layer = nn.Linear(128, 10)

    def forward(self, x):
        layer = self.layer(x)
        con_layer = layer.view(-1, 1 * 1 * 512)
        feature_out = self.feature_layer(con_layer)
        cls_out = self.cls_layer(feature_out)
        return feature_out, cls_out


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.layers = make_layers(cfg['E'])
        self.features = nn.Sequential(
            nn.Linear(512, 2)
        )
        self.classfier = nn.Sequential(
            nn.Linear(2, 10)
        )

        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        feature_out = self.features(out)
        cls_out = self.classfier(feature_out)
        return feature_out, cls_out


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


if __name__ == '__main__':
    x = torch.randn(1024, 3, 28, 28)
    net = Net()
    start_time = time.time()
    out = net(x)
    end_time = time.time()
    use_time = end_time - start_time
    print("use_time: ", use_time)
    print(out.size())
    print(sum(param.numel() for param in net.parameters()))
