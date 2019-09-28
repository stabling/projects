import torch.nn as nn
import torch
from extend.conv import Conv2dLocal


class MainNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            Conv2dLocal(22, 26, 48, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            Conv2dLocal(10, 12, 64, 128, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.local_layer = nn.Sequential(
            Conv2dLocal(4, 5, 128, 128, 3, 1, 1)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256*4*5, 1024),
            nn.Linear(1024, 3000)
        )

    def forward(self, x):
        print("fc_layer: ", self.fc_layer)
        layer1 = self.layers(x)
        layer2 = self.local_layer(layer1)
        con_layer = torch.cat((layer1, layer2), dim=1)
        con_layer = con_layer.view(-1, 256*4*5)
        out = self.fc_layer(con_layer)
        return out


if __name__ == '__main__':
    x = torch.randn(5, 3, 96, 112)
    net = MainNet()
    out = net(x)
    print(out)