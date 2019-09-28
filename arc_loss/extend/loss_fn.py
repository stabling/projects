import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import argparse


def center_loss(feature_out, net, y):
    """
    center_loss implement
    :param feature_out: 特征层的输出
    :param net: 实例化的网络
    :param y: 标签
    :return:
    """
    center = net.cls_layer.weight
    center_exp = center.index_select(dim=0, index=y.long())
    count = torch.histc(y, bins=len(center), min=0, max=len(center) - 1)
    count_dis = count.index_select(dim=0, index=y.long())
    return torch.mean(torch.sum((feature_out - center_exp) ** 2, dim=1) / count_dis.float())


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)).cuda())
        self.func = nn.Softmax()

    def forward(self, x, s, m):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax


if __name__ == '__main__':
    b = torch.randn(5, 10)
