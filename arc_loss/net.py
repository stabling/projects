import torch.nn as nn
import torch
from extend.loss_fn import Arcsoftmax


class MainNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 1),
        )

        self.arcsoftmax = Arcsoftmax(2, 10)

        self.feature_layer = nn.Sequential(
            nn.Linear(1*1*64, 2)
        )

        self.cls_layer = nn.Linear(2, 10)

    def forward(self, x, s, m):
        layer = self.layers(x)
        con_layer = layer.view(-1, 1*1*64)
        feature_out = self.feature_layer(con_layer)
        cls_out = self.arcsoftmax(feature_out, s, m)
        cls_out = torch.log(cls_out)
        return feature_out, cls_out


if __name__ == '__main__':
    a = torch.randn(64, 3, 56, 56)
    net = MainNet()
    b = net(a)
    print(b)