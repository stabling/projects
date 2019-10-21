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
        with open("/home/yzs/myproject/yolo-v3/label_.txt") as f:
            self.dataset = f.readlines()

    def __getitem__(self, idx):
        labels = {}
        line = self.dataset[idx]
        strs = line.strip().split()
        _img_data = Image.open(os.path.join(cfg.IMG_BASE_DIR, strs[0]))
        img_data = cfg.TRANSFORM(_img_data)
        for feature_size, anchors in cfg.ANCHOR_GROUPS.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            params = self.dataset[idx].split(" ")
            box = np.array([float(x) for x in params[1:6]])
            cls, cx, cy, w, h = box
            cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMAGE_WIDTH)
            cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMAGE_HEIGHT)

            for i, anchor in enumerate(anchors):
                iou = utils.iou_loss(w, h, anchor)
                p_w, p_h = w / anchor[0], h / anchor[1]
                labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                    [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *utils.one_hot(cfg.CLASS_NUM, int(cls))])

        return torch.Tensor(labels[13]), torch.Tensor(labels[26]), torch.Tensor(labels[52]), img_data

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    mydata = MyDataset()
    print(mydata[2])
