import torch, os
from torch import nn, optim
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.nn import init, functional as F
from torch import nn
import random
import time


def createMask(pos_num, esp=0):
    m = torch.empty(1, 1, pos_num, pos_num).fill_(esp)
    for i in range(pos_num):
        for j in range(i + 1):
            m[:, :, i, j] = 1

    return m


if __name__ == '__main__':
    w = torch.randn(1, 1, 5, 5)
    mask = createMask(5)
    print(w)
    print(F.normalize(torch.softmax(w, dim=-1) * mask, p=1, dim=-1))
