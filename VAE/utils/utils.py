import torch.nn as nn
import torch
from thop import profile, clever_format

"""
the library of base classes that have been defined(已经定义好的基础类仓库)
Contain(包含):
-----base convolution(基本卷积层)
-----residual layer(基本的残差层)
-----pool layer(基本的池化层)
-----upsample layer(基本的上采样层)
-----initialization parameters(初始化参数)
-----count the flops and params(统计计算量和参数量)
"""


# Define a base convolution(定义基本的卷积层)
class ConvolutionLayer(nn.Module):

    def __init__(self, list, act_func=nn.ReLU(True)):
        """
        Input Parameters(输入参数)-----
        :param list: convolution parameters list(卷积参数列表)
        :param act_func: activation function(使用的激活函数)
        """
        super().__init__()
        self.base_conv = nn.Sequential(
            nn.Conv2d(*list),
            nn.BatchNorm2d(list[1]),
            act_func
        )

    def forward(self, x):
        return self.base_conv(x)


# Define a base transpose convolution(定义基本的转置卷积层)
class TransposeConvolutionLayer(nn.Module):

    def __init__(self, list, act_func=nn.ReLU(True)):
        """
        Input Parameters(输入参数)-----
        :param list: convolution parameters list(转置卷积参数列表)
        :param act_func: activation function(使用的激活函数)
        """
        super().__init__()
        self.base_conv = nn.Sequential(
            nn.ConvTranspose2d(*list),
            nn.BatchNorm2d(list[1]),
            act_func
        )

    def forward(self, x):
        return self.base_conv(x)


# Define a residual layer(定义基本的残差类)
class ResidualLayer(nn.Module):

    def __init__(self, in_channel, times=1):
        """
        Input Parameters(输入参数)-----
        :param in_channel: Input channel(输入通道)
        :param times: the number of residual layer(残差的层数)
        """
        super().__init__()
        self.res = []
        for i in range(times):
            self.res += [ConvolutionLayer([in_channel, in_channel * 2, 1, 1, 0]),
                         ConvolutionLayer([in_channel * 2, in_channel, 3, 1, 1])]
        self.out = nn.Sequential(*self.res)

    def forward(self, x):
        return self.out(x) + x


# Define a pool layer(定义基本的池化类)
class PoolLayer(nn.Module):

    def __init__(self, kernel_size, stride, mode="max"):
        """
        Input Parameters(输入参数)-----
        :param kernel_size: kernel_size(卷积核大小)
        :param stride: stride(步长)
        :param mode: pool mode(池化方式)
        """
        super().__init__()
        if mode == "max":
            self.down = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride)
            )
        elif mode == "mean":
            self.down = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride)
            )

    def forward(self, x):
        return self.down(x)


# Define an upsample layer(定义一个基本的上采样层)
class UpsampleLayer(nn.Module):

    def __init__(self, scale_factor, mode="bilinear"):
        """
        Input Parameters(输入参数)-----
        :param scale_factor: scale factor(比例因子)
        :param mode: upsample mode(上采样方式)
        """
        super().__init__()
        if mode == "bilinear":
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        elif mode == "nearest":
            self.up = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x):
        return self.up(x)


# Set the initialization parameters(设置初始化参数)
class init_weight:

    def __init__(self, module_name, module_type, mode):
        """
        Input Parameters(输入参数)-----
        :param module_name: module name, for example:self.conv(模块名称)
        :param module_type: The module type of the initialization parameter list,
        for example:[nn.Conv2d,](初始化参数的模块类型)
        :param mode: initialization mode list, for example:[nn.init.normal_,]
        (初始化方式列表，module_type和mode的输入列表长度必须相等)
        """
        super(init_weight, self).__init__()
        self.module_name = module_name
        self.module_type = module_type
        self.mode = mode

    def init_weight(self):
        for i in range(len(self.module_type)):
            for layer in self.module_name.modules():
                if isinstance(layer, self.module_type[i]):
                    param_shape = layer.weight.shape
                    w = torch.empty(param_shape)
                    self.mode[i](w)


# count the flops and params(统计浮点计算量和参数)
class Get_Consume:

    def __init__(self, net, input):
        """
        Input Parameters(输入参数)-----
        :param net: init the net model instance(初始网络模型实例)
        :param input: input data, for example:(input, )(输入的数据，例如：(input, ))
        """
        self.net = net
        self.input = input

    def get_consume(self):
        flops, params = profile(self.net, inputs=self.input)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
