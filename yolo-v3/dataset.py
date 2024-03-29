from torch.utils.data import Dataset
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((416, 416)),
    torchvision.transforms.ToTensor()
])


class MyDataset(Dataset):

    def __init__(self):
        with open(cfg.LABEL_FILE) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        _img_data = Image.open(os.path.join(cfg.IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)

        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 6), dtype=np.float32)

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)

                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    box_area = w * h

                    # 计算置信度(同心框的IOU(交并))
                    inter = np.minimum(w, anchor[0]) * np.minimum(h, anchor[1])  # 交集
                    conf = inter / (box_area + anchor_area - inter)

                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [cx_offset, cy_offset, np.log(p_w), np.log(p_h), conf, int(cls)])

        return labels[13], labels[26], labels[52], img_data
