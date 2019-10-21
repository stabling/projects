import torch
from torch.utils.data import Dataset
import numpy as np
import utils
import cfg
import os

from PIL import Image
import math


class MyDataset(Dataset):

    def __init__(self):
        with open(cfg.LABEL_FILE) as f:
            self.dataset = f.readlines()

    def __getitem__(self, x):
        labels = {}
        line = self.dataset[x]
        strs = line.strip().split()
        _img_data = Image.open(os.path.join(cfg.IMG_BASE_DIR, strs[0]))
        img_data = cfg.TRANSFORM(_img_data)

        _boxes = np.array([float(x) for x in strs[1:]])  # 将除文件名后的标签数据取出
        boxes = np.split(_boxes, len(_boxes) // 5)  # 将标签分割成不同的对象

        for feature_size, anchors in cfg.ANCHOR_GROUPS.items():  # 将尺寸大小和锚框分别遍历出来
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))  # 定义标签的形状

            for box in boxes:  # 遍历各标签
                cx, cy, w, h, cls = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMAGE_WIDTH)  # 得到x的偏移量和索引值
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMAGE_HEIGHT)  # 得到y的偏移量和索引值

                for i, anchor in enumerate(anchors):  # 遍历锚框
                    iou = utils.iou_loss(w, h, anchor)  # api: 使用iou损失
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *utils.one_hot(cfg.CLASS_NUM, int(cls))])
                    # p_w(p_h) = log(w(h) / anchor[0](anchor[1]))  shape:(cy_index, cx_index, anchor, iou+gt+category)

        return torch.Tensor(labels[13]), torch.Tensor(labels[26]), torch.Tensor(labels[52]), img_data

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    mydata = MyDataset()
    print(mydata[2])
