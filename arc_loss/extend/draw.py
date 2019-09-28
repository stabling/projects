import matplotlib.pyplot as plt
from extend.config import color
import torch


def draw(feature_out, output):
    """

    :param feature_out: 输出的特征点
    :param output: 分类输出结果
    :return:
    """
    plt.clf()
    a = torch.argmax(output, dim=1)
    b = feature_out
    for i in range(10):
        index_select = torch.nonzero(a == i).view(-1)
        out = b[index_select]
        plt.scatter(out[:, 0], out[:, 1], c=color[i], marker=".")
    plt.pause(0.01)


if __name__ == '__main__':
    for i in range(10):
        a = torch.randn(512, 10)
        b = torch.randn(512, 2)
        draw(b, a)

