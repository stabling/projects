import torch
from torch.utils.data import Dataset
import numpy as np
import cfg
import os
from torchvision import transforms

from PIL import Image
import math

LABEL_FILE_PATH = r"/home/yzs/dataset/tiny_coco/Anno/Anno_list.txt"
IMG_BASE_DIR = r"/home/yzs/dataset/tiny_coco"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __getitem__(self, x):
        labels = {}
        line = self.dataset[x]
        strs = line.strip().split()
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transform(_img_data)

        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHOR_GROUPS.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMAGE_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMAGE_HEIGHT)

                for i, anchor in enumerate(anchors):
                    anchor_area = anchor[0] * anchor[1]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    inter_area = min(w, anchor[0]) * min(h, anchor[1])
                    iou = inter_area / (p_area + anchor_area - inter_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])

        return torch.Tensor(labels[13]), torch.Tensor(labels[26]), torch.Tensor(labels[52]), img_data

    def __len__(self):
        return len(self.dataset)
