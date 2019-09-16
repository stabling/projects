import numpy as np
# import Augmentor
from PIL import Image
import re
import linecache
import torch


def iou(box, boxes, isMin=False):
    """
    计算iou的大小
    :param box: 传入的标签(cls, x1, y1, x2, y2)
    :param boxes: 要比较iou的box列表
    :param isMin: 要使用的mode
    :return:
    """

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)

    if isMin == True:
        iou = inter_area / np.minimum(box_area, boxes_area)
    else:
        iou = inter_area / (box_area + boxes_area - inter_area)

    return iou


def nms(boxes, threshold, isMin=False):
    """
    非极大值抑制的实现
    :param boxes: 传入的标签(cls, x1, y1, x2, y2)
    :param threshold: 传入的阈值
    :param isMin: iou的mode
    :return:
    """

    sort_boxes = boxes[np.argsort(-boxes[:, 0])]
    save_boxes = []
    while len(sort_boxes) > 1:
        box = sort_boxes[0]
        save_boxes.append(box)

        boxes_ = sort_boxes[1:]
        sort_boxes = boxes_[iou(box, boxes_, isMin) < threshold]

    if len(sort_boxes) > 0:
        save_boxes.append(sort_boxes[0])

    return np.stack(save_boxes)


def save_file(filename, list):
    """
    save log file
    :param filename: log文件的名字
    :param list: 要保存参数的列表
    :return:
    """
    with open(filename, "a") as f:
        for param in list:
            f.write(param)
            f.write(" ")
        f.write("\n")
        f.flush()


def center_gen_pic(box, epoch, rate):
    """
    在标签中心点附近生成新的坐标点
    :param box: 传入的标签(x1, y1, x2, y2)
    :param epoch: 循环的次数
    :param rate: 传入各比率的数组
    :return:
    """
    center = np.array([0.5 * (box[i] + box[i + 2]) for i in range(2)])
    wh = np.array([box[i + 2] - box[i] for i in range(2)], dtype=float)
    new_center_ = np.array(
        [center[i] + np.random.randint(-rate[0] * wh[i], rate[0] * wh[i] + 1, size=(epoch, 1)) for i in range(2)])
    new_center = np.concatenate([new_center_[0], new_center_[1]], axis=1)

    side_len = np.random.randint(np.min(wh) * rate[1], np.max(wh) * rate[2] + 1, size=(epoch, 1))
    box_ = np.tile(box, (epoch, 1))

    box_[:, 0] = new_center[:, 0] - 0.5 * side_len[:, 0]
    box_[:, 1] = new_center[:, 1] - 0.5 * side_len[:, 0]
    box_[:, 2] = box_[:, 0] + side_len[:, 0]
    box_[:, 3] = box_[:, 1] + side_len[:, 0]
    boxes = box_[:, :4]

    new_box = (box - np.tile(new_center, (7))) / np.tile(side_len, (14))

    return boxes, new_box


def round_gen_pic(weight, height, size, epoch, rate):
    """
    在图片的周围生成新的坐标点
    :param weight: 原图片的宽
    :param height: 原图片的高
    :param size: 最小的尺寸
    :param epoch: 循环的次数
    :param rate: 传入各比率的数组
    :return:
    """
    center_x = np.random.randint(rate[0] * weight, rate[1] * weight + 1, size=(epoch, 1))
    center_y = np.random.randint(rate[0] * weight, rate[1] * weight + 1, size=(epoch, 1))

    side_len = np.random.randint(min(size, min(weight, height)), max(size, min(weight, height) * rate[2]) + 1,
                                 size=(epoch, 1))

    boxes = np.concatenate(
        [center_x - 0.5 * side_len[:], center_y - 0.5 * side_len[:], center_x + 0.5 * side_len[:],
         center_y + 0.5 * side_len[:]], axis=1)

    return boxes[:, :4]


def widerface_change_txt(file_path, save_path):
    """
    将widerface里的标签转换成txt文本,格式为(filename, x1, y1, x2, y2)
    :param file_path: 标签的路径
    :param save_path: 保存的路径
    :return:
    """

    with open(save_path, "w") as f1:
        for i, line in enumerate(open(file_path, "r").readlines()):
            if re.search('jpg', line):
                file_name = line.strip().split('\\')
                i += 2
                face_count = int(linecache.getline(file_path, i))

                for j in range(face_count):
                    box_line = linecache.getline(file_path, i + j + 1)
                    strs = box_line.strip().split()
                    x1, y1, w, h = int(strs[0]), int(strs[1]), int(strs[2]), int(strs[3])

                    x2 = x1 + w
                    y2 = y1 + h

                    f1.write(str(file_name[0]) + " ")
                    f1.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")

    f1.close()


def widerface_neg_txt(file_path, save_path):
    """
    生成负样本必需的标签文本
    :param file_path: 文本的路径
    :param save_path: 保存的路径
    :return:
    """
    with open(save_path, "w") as f1:
        for i, line in enumerate(open(file_path, "r").readlines()):
            if re.search('jpg', line):
                file_name = line.strip().split('\\')
                i += 2
                face_count = int(linecache.getline(file_path, i))
                f1.write(str(file_name[0]) + " ")

                for j in range(face_count):
                    box_line = linecache.getline(file_path, i + j + 1)
                    strs = box_line.strip().split()
                    x1, y1, w, h = int(strs[0]), int(strs[1]), int(strs[2]), int(strs[3])

                    x2 = x1 + w
                    y2 = y1 + h

                    f1.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " ")
                f1.write("\n")

    f1.close()


def convert_to_square(box, image):
    """
    将框转换成正方形
    :param box: 传入的标签(x1, y1, x2, y2)
    :return:
    """

    square_box = box.copy()
    if len(box) == 0:
        return np.array([])
    boxes = box[:, 1:5]
    for box_ in boxes:
        _x1 = int(box_[0])
        _y1 = int(box_[1])
        _x2 = int(box_[2])
        _y2 = int(box_[3])

    w = square_box[:, 3] - square_box[:, 1]
    h = square_box[:, 4] - square_box[:, 2]
    max_side = np.maximum(w, h)
    center_x = square_box[:, 1] + 0.5 * w
    center_y = square_box[:, 2] + 0.5 * h
    square_box[:, 1] = center_x - max_side * 0.5
    square_box[:, 2] = center_y - max_side * 0.5
    square_box[:, 3] = square_box[:, 1] + max_side
    square_box[:, 4] = square_box[:, 2] + max_side

    return square_box[:, 1:5]


# 批量反算
def _box(index, offset, landmark, cls, scale, stride=2, side_len=12):
    """
    将offset对应回原图
    :param index: 边框对应的索引
    :param offset: 偏移量
    :param landmark: 关键点
    :param cls: 置信度
    :param scale: 缩放比例
    :param stride: 步长
    :param side_len: 最小边长
    :return:
    """
    _offset = offset[:, index[:, 0], index[:, 1]].transpose(1, 0)
    _landmark = landmark[:, index[:, 0], index[:, 1]].transpose(1, 0)

    center = np.array(
        [(np.array(index[:, i], dtype=float) * stride + 0.5 * side_len) / scale for i in [1, 0]]).transpose(1, 0)
    wh = np.tile(np.array([side_len / scale, side_len / scale]), (1, 1))
    offset = np.array(_offset) * np.tile(wh, (1, 2)) + np.tile(center, (1, 2))
    landmark = np.array(_landmark) * np.tile(wh, (1, 5)) + np.tile(center, (1, 5))
    cls = np.array(cls).reshape(-1, 1)

    bboxes = torch.tensor(np.concatenate([cls, offset, landmark], axis=1))

    return bboxes


if __name__ == '__main__':
    a = np.array([[3, 3, 6, 8], [4, 4, 7, 10], [7, 7, 9, 12]])
    b = convert_to_square(a)
    print(b)
