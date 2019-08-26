import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["face"]  # 脸部检测


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(i):
    in_file = open(r'D:\project\project2.0\yolo-v3\xml\%s.xml' % (i))

    out_file = open(r'D:\project\project2.0\yolo-v3\label\label.txt', 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


image_ids_train = open('/home/*****/darknet/scripts/VOCdevkit/voc/list').read().strip().split()  # list格式只有000000 000001

# image_ids_val = open('/home/*****/darknet/scripts/VOCdevkit/voc/list').read().strip().split()


list_file_train = open('boat_train.txt', 'w')
list_file_val = open('boat_val.txt', 'w')

for image_id in image_ids_train:
    list_file_train.write('/home/*****/darknet/scripts/VOCdevkit/voc/JPEGImages/%s.jpg\n' % (image_id))
    convert_annotation(image_id)
list_file_train.close()  # 只生成训练集，自己根据自己情况决定