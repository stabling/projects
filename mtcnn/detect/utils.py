"""utilities for the joint face detection and alignment tasks using MTCNN. """

import torch
from test.data_process import SAMPLE_ID


def est_measure(outputs, labels, thres):
    outputs = outputs.reshape(-1)

    p_mask, n_mask = torch.eq(labels, SAMPLE_ID['positive']), torch.eq(labels, SAMPLE_ID['negative'])
    p, n = p_mask.sum(), n_mask.sum()

    tp = torch.gt(outputs[p_mask], 1 - thres).sum()
    tn = torch.lt(outputs[n_mask], thres).sum()

    return torch.stack([tp + tn, tp, p + n, p])


def pyramid_feature_mapping(indexes, scale, stride=2, side_len=12):
    """feature map back calculation on image pyramid. """
    xy1 = indexes[:, [1, 0]] * stride
    xy2 = xy1 + side_len

    _boxes = torch.cat([xy1, xy2], dim=1)
    return torch.ceil(_boxes.float() / scale)


def box_regression(bounding_boxes, offset):
    """calculate the actual boxes by offset to bounding boxes. """
    _boxes = bounding_boxes.float()
    w, h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
    ox1, oy1, ox2, oy2 = offset[:, 0], offset[:, 1], offset[:, 2], offset[:, 3]

    spacing = torch.stack([ox1 * w, oy1 * h, ox2 * w, oy2 * h], dim=1)
    box = torch.add(_boxes, spacing)
    return torch.round(box).int()


def cord_regression(bounding_boxes, offset):
    """calculate the actual points coordinates by offset to bounding boxes' centers. """
    _boxes = bounding_boxes.float()
    side_lens = _boxes[:, 2:] - _boxes[:, :2]
    cxy = (_boxes[:, :2] + _boxes[:, 2:]) / 2

    m = offset.size(1) // 2
    pos = torch.add(cxy.repeat((1, m)), torch.mul(offset, side_lens.repeat((1, m))))
    return torch.round(pos).int()


def convert_to_square(boxes):
    _boxes = boxes.float()
    w, h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]

    max_side_len = torch.max(w, h).reshape(-1, 1)
    cxy = torch.stack([_boxes[:, 0] + _boxes[:, 2], _boxes[:, 1] + _boxes[:, 3]], dim=1)

    square = torch.cat([cxy - max_side_len, cxy + max_side_len], dim=1) / 2
    return square.int()


def nms(boxes, confidences, threshold=0.3, mode='union'):
    """Non-Maximum suppression, greedily accepting local maximal and discarding
       their neighbours.
    """
    if boxes.size(0) == 0:
        return []
    boxes = boxes.float()

    areas = torch.mul(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    idxs = torch.argsort(confidences)

    keep = []
    while idxs.size(0) > 0:
        last = idxs[-1]
        keep.append(last)

        luc = torch.max(boxes[last][:2], boxes[idxs[:-1], :2])
        rlc = torch.min(boxes[last][2:], boxes[idxs[:-1], 2:])
        inters_area = torch.prod(torch.max(rlc - luc, torch.tensor(0.)), dim=1)

        if mode == 'union':
            iou = inters_area / (areas[last] + areas[idxs[: -1]] - inters_area)
        elif mode == 'minimum':
            iou = inters_area / torch.min(areas[last], areas[idxs[:-1]])
        else:
            raise RuntimeError("invalid mode: excepted mode 'minimum' or 'union', but get '{}'.".format(mode))

        idxs = idxs[:-1][iou < threshold]

    return torch.stack(keep)
