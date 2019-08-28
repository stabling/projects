import torch.nn as nn
import torch
from utils.utils import ConvolutionLayer, TransposeConvolutionLayer


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvolutionLayer([1, 64, 3, 2, 1]),
            ConvolutionLayer([64, 128, 3, 2, 1]),
            ConvolutionLayer([128, 256, 3, 2, 0]),
            ConvolutionLayer([256, 512, 3, 1, 0]),
            ConvolutionLayer([512, 256, 1, 1, 0]),
            nn.Conv2d(256, 2, 1, 1, 0)
        )

    def forward(self, x):
        out = self.conv(x)
        logsigma = out[:, :1, :, :]
        miu = out[:, 1:, :, :]
        return logsigma, miu


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeConvolutionLayer([128, 256, 1, 1, 0]),
            TransposeConvolutionLayer([256, 512, 1, 1, 0]),
            TransposeConvolutionLayer([512, 256, 3, 1, 0]),
            TransposeConvolutionLayer([256, 128, 3, 2, 0]),
            TransposeConvolutionLayer([128, 64, 3, 2, 1, 1]),
            TransposeConvolutionLayer([64, 1, 3, 2, 1, 1]),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        logsigma, miu = self.encoder(x)
        z = torch.randn(128).cuda()
        z = z * torch.exp(logsigma) + miu
        z = z.permute(0, 3, 1, 2)
        decoder_out = self.decoder(z)
        return decoder_out, logsigma, miu


if __name__ == '__main__':
    input = torch.randn(10, 1, 28, 28)
    net = MainNet()
    out = net(input)
    print(out.size())