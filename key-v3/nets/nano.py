import torch.nn as nn


class DWConv(nn.Module):

    def __init__(self, in_channels, mode="normal"):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        if mode == "normal":
            self.wconv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(True)
            )
        elif mode == "reduce":
            self.wconv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(True)
            )
        self.bconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        out_1 = self.dconv(x)
        out_2 = self.wconv(out_1)
        out = self.bconv(out_2)
        return out


class EP(nn.Module):

    def __init__(self, in_channels, mode):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(True),
            DWConv(in_channels * 2, mode),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.mode = mode

    def forward(self, x):
        if self.mode == "normal":
            out = self.conv(x) + x
        elif self.mode == "reduce":
            out = self.conv(x)
        return out


class PEP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels * 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(True),
            DWConv(in_channels * 2, "normal"),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        if self.in_channels == self.out_channels:
            out = self.conv(x) + x
        else:
            out = self.conv(x)
        return out


class FCA(nn.Module):

    def __init__(self, in_channels, out_channels_1, out_channels_2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels_1),
            nn.ReLU(True),
            nn.Linear(out_channels_1, out_channels_2),
            nn.Sigmoid()
        )

    def forward(self, x):
        layer_in = x.reshape(-1, 52 * 52 * 150)
        layer_out = self.fc(layer_in)
        ratio = layer_out.reshape(-1, 150, 1, 1)
        out = x * ratio
        return out


class BaseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mode):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding),
        self.activate = nn.ReLU(True)
        self.mode = mode

    def forward(self, x):
        out_1 = self.conv
        if self.mode == "out":
            out = out_1
        elif self.mode == "normal":
            out = self.activate(out_1)
        return out


class MainNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.BASE_52 = nn.Sequential(
            BaseConv(3, 12, 3, 2, 1, "normal"),
            BaseConv(12, 24, 3, 1, 1, "normal"),
            PEP(24, 7, 70),
            EP(70, "reduce"),
            PEP(70, 25, 70),
            PEP(70, 24, 150),
            EP(150, "reduce"),
            PEP(150, 56, 150),
            BaseConv(150, 150, 1, 1, 0, "normal"),
            FCA(52 * 52 * 150, 8, 150),
            PEP(150, 73, 150),
            PEP(150, 71, 150),
            PEP(150, 75, 325)
        )
        self.BASE_26 = nn.Sequential(
            EP(325, "reduce"),
            PEP(325, 132, 325),
            PEP(325, 124, 325),
            PEP(325, 141, 325),
            PEP(325, 140, 325),
            PEP(325, 137, 325),
            PEP(325, 135, 325),
            PEP(325, 133, 325),
            PEP(325, 124, 545)
        )
        self.BASE_13 = nn.Sequential(
            EP(545, "reduce"),
            PEP(545, 276, 545),
            BaseConv(545, 230, 1, 1, 0, "normal"),
            EP(230, "normal"),
            PEP(489, 213, 469),
            BaseConv(469, 189, 1, 1, 0, "normal")
        )
        self.DETECT_52 = nn.Sequential(
            PEP()
        )

    def forward(self, x):
        pass
