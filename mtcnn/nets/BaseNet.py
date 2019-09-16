import torch.nn as nn
import torch
import torch.nn.functional as F


class PNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )

        self.cls_layer = nn.Conv2d(32, 1, 1, 1)

        self.off_layer = nn.Conv2d(32, 4, 1, 1)

        self.land_layer = nn.Conv2d(32, 10, 1, 1)

    def forward(self, x):
        out = self.layer(x)
        cls = F.sigmoid(self.cls_layer(out))
        offset = self.off_layer(out)
        landmark = self.land_layer(out)

        return cls, offset, landmark


class RNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(24),
            nn.Conv2d(3, 28, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(28, 48, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU()
        )

        self.linear_layer = nn.Linear(3 * 3 * 64, 128)

        self.cls_layer = nn.Linear(128, 1)

        self.off_layer = nn.Linear(128, 4)

        self.land_layer = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 3 * 3 * 64)
        out = self.linear_layer(out)
        cls = F.sigmoid(self.cls_layer(out))
        offset = self.off_layer(out)
        landmark = self.land_layer(out)

        return cls, offset, landmark


class ONet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(48),
            nn.Conv2d(3, 32, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU()
        )

        self.linear_layer = nn.Linear(3 * 3 * 128, 256)

        self.cls_layer = nn.Linear(256, 1)

        self.off_layer = nn.Linear(256, 4)

        self.land_layer = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 3 * 3 * 128)
        out = self.linear_layer(out)
        cls = F.sigmoid(self.cls_layer(out))
        offset = self.off_layer(out)
        landmark = self.land_layer(out)

        return cls, offset, landmark


if __name__ == '__main__':
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    # print(onet)
    a = torch.rand(size=(1, 3, 409, 687))
    b = torch.rand(size=(1, 3, 24, 24))
    c = torch.rand(size=(1, 3, 48, 48))
    cls_a, offset_a, landmark_a = pnet(a)
    cls_b, offset_b, landmark_b = rnet(b)
    cls_c, offset_c, landmark_c = onet(c)
    print("a: ", cls_a.size(), offset_a.size(), landmark_a.size())
    print("b: ", cls_b.size(), offset_b.size(), landmark_b.size())
    print("c: ", cls_c.size(), offset_c.size(), landmark_c.size())
    nn.AdaptiveAvgPool2d()
