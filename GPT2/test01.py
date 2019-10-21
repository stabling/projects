import torch
from torch import nn


class CBOW(nn.Module):

    def __init__(self, voc_num, voc_dim):
        super().__init__()

        self.codebook = nn.Embedding(voc_num, voc_dim)
        # self.codebook = nn.Parameter(torch.randn(voc_num,voc_dim))

        self.linear_1 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_2 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_4 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_5 = nn.Linear(voc_dim, voc_dim, bias=False)

    def forward(self, x1, x2, x4, x5):
        v1 = self.codebook(x1)
        v2 = self.codebook(x2)
        v4 = self.codebook(x4)
        v5 = self.codebook(x5)

        y1 = self.linear_1(v1)
        y2 = self.linear_1(v2)
        y4 = self.linear_1(v4)
        y5 = self.linear_1(v5)

        return y1 + y2 + y4 + y5

    def getLoss(self, x3, y3):
        v3 = self.codebook(x3)

        return torch.mean((y3 - v3) ** 2)
