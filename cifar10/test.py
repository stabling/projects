import torch
import numpy as np
from PIL import Image
from thop import profile, clever_format


# Required to validate(有待验证)
# Implement the visualization of features(实现了特征的可视化)
class FeatureVisualization():
    def __init__(self, input, net, path, selected_layer):
        """
        Input Parameters(输入参数)-----
        :param input: input data(输入的数据)
        :param net: net(网络模型)
        :param path: the save path of parameters(参数的保存路径)
        :param selected_layer: the visualization layer of the features that you select,
        for example:7(选择的可视化层,例如：第七层)
        """
        self.input = input
        self.net = net.load_state_dict(torch.load(path))
        self.selected_layer = selected_layer

    # Get the features of all channels(得到所有通道的特征)
    def get_feature(self):
        for index, layer in enumerate(self.net):
            x = layer(self.input)
            if index == self.selected_layer:
                return x

    # Get the feature that only on the first channel(仅得到第一个通道的特征)
    def get_single_feature(self):
        features = self.get_feature()
        feature = features[:, 0, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        return feature

    # Convert from feature into picture, mode: "L"(将特征转换成图片，模式：灰度图)
    def save_feature_to_img(self):
        feature = self.get_single_feature()
        feature = feature.data.numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.array(np.round(feature * 255), dtype=np.uint8)
        img = Image.fromarray(feature, mode="L")
        img.save("./feature.jpg")


class Get_Consume():

    def __init__(self, net, input):
        self.net = net
        self.input = input

    def get_consume(self):
        flops, params = profile(self.net, inputs=self.input)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
