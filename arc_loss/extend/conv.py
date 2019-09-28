import torch.nn as nn
import torch
import math
from extend.modules.utils import _pair
import torch.nn.functional as F


class Conv2dLocal(nn.Module):

    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = nn.Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return F.conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


if __name__ == '__main__':
    a = torch.randn(5, 3, 96, 112)
