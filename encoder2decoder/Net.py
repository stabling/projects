import torch.nn as nn
from utils.utils import ConvolutionLayer, TransposeConvolutionLayer


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvolutionLayer([3, 32, 3, 1]),
            ConvolutionLayer([32, 64, 3, 1]),
            ConvolutionLayer([64, 84, 3, 1]),
            ConvolutionLayer([84, 128, 3, 1]),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeConvolutionLayer([128, 84, 3, 1]),
            TransposeConvolutionLayer([84, 64, 3, 1]),
            TransposeConvolutionLayer([64, 32, 3, 1]),
            TransposeConvolutionLayer([32, 3, 3, 1]),
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
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out
