import torch
import numpy as np


def nms(boxes, thresh):
    """
    nms的基础模块
    :param boxes: 用来做nms的边框, (x1, y1, x2, y2, cls, category)
    :param thresh: nms的阈值
    :return:
    """
    sort_boxes = boxes[np.argsort(-boxes[:, 4])]
    keep_boxes = []
    while len(sort_boxes) > 1:
        box = sort_boxes[0]
        keep_boxes.append(box)
        boxes = sort_boxes[1:]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter = w * h

        iou = np.true_divide(inter, (box_area + area - inter))

        sort_boxes = boxes[np.where(iou < thresh)]

    if len(sort_boxes) > 0:
        keep_boxes.append(sort_boxes[0])

    return keep_boxes


def catergory(boxes, thresh):
    """
    根据类别进行mns
    :param boxes: 用来做nms的边框, (x1, y1, x2, y2, cls, category)
    :param thresh: nms的阈值
    :return:
    """
    boxes = np.array(boxes)
    keep_boxes = []

    while len(boxes) > 1:
        _box = boxes[0]
        same_boxes = (boxes[_box[5] == boxes[:, 5]])
        keep_boxes.extend(nms(same_boxes, thresh))
        boxes = boxes[_box[5] != boxes[:, 5]]
    if len(boxes) > 0:
        keep_boxes.append(boxes[0])

    keep_boxes = torch.Tensor(np.stack(keep_boxes))

    return keep_boxes


def one_hot(cls_num, v):
    """
    转换成独热编码模式
    :param cls_num: 类别数
    :param v: 值
    :return:
    """
    b = np.zeros(cls_num)
    b[v] = 1
    return b


def iou_loss(w, h, anchor):
    """
    求iou的损失
    :param w: 标签的宽
    :param h: 标签的高
    :param anchor: 锚框
    :return:
    """
    anchor_area = anchor[0] * anchor[1]
    p_area = w * h
    inter_area = min(w, anchor[0]) * min(h, anchor[1])
    iou = inter_area / (p_area + anchor_area - inter_area)
    return iou


def loss_func(output, target, alpha):
    """
    定义yolo的损失函数
    :param output: 输出
    :param target: 标签
    :param alpha: 系数
    :return:
    """
    output = output.permute(0, 2, 3, 1)
    output = output.view(output.size(0), output.size(1), output.size(2), 3, -1)
    obj = target[..., 0] > 0
    noobj = target[..., 0] == 0

    obj_loss = torch.mean((output[obj] - target[obj]) ** 2)
    noobj_loss = torch.mean((output[noobj] - target[noobj]) ** 2)

    loss = alpha * obj_loss + (1 - alpha) * noobj_loss

    return loss
