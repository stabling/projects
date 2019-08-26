import torch
import numpy as np


def nms(boxes, thresh):
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
