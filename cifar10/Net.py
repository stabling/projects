import torch.nn as nn
from utils.utils import ConvolutionLayer, PoolLayer, ResidualLayer, init_weight
import torch


# Main module, need to rewrite each time(主模块，每次需要重写)
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.pre_conv = nn.Sequential(
            ConvolutionLayer([3, 64, 3, 1, 1]),
            ConvolutionLayer([64, 64, 3, 2, 0]),
            ResidualLayer(64),
            ResidualLayer(64),
            ConvolutionLayer([64, 64, 3, 2, 0]),
            ConvolutionLayer([64, 128, 3, 1, 1]),
            ResidualLayer(128),
            ResidualLayer(128),
            ConvolutionLayer([128, 128, 3, 2, 0]),
            ConvolutionLayer([128, 256, 3, 1, 1]),
            ResidualLayer(256),
            ResidualLayer(256),
            ConvolutionLayer([256, 256, 3, 2, 0]),
            ConvolutionLayer([256, 256, 3, 1, 1]),
        )
        self.layer = nn.Sequential(
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.pre_conv(x)
        out = out.reshape(-1, 256 * 1 * 1)
        out = self.layer(out)
        return out


if __name__ == '__main__':
    a = torch.randn(256, 3, 32, 32)
    net = MainNet()
    out = net(a)
