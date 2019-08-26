from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
import os
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir = r'E:\BaiduNetdiskDownload'
dataType = 'val2017'
annFile = r'{}\annotations\instances_{}.json'.format(dataDir, dataType)

valDir = r"E:\BaiduNetdiskDownload\val2017"
newDir = r"D:\project\project2.0\yolo-v3\coco_pro\new_pic"
# 初始化标注数据的 COCO api
coco = COCO(annFile)

# _img = coco.getImgIds(catIds=2)
# len_label = 12
# for i, index in enumerate(_img):
#     strs = (len_label - len(str(index)))*"0" + str(index) + ".jpg"
#     print(strs)
#     img_path = os.path.join(valDir, strs)
#     img = Image.open(img_path)
#     img.save(os.path.join(newDir, "{}.jpg".format(i)))

_anno = coco.getAnnIds(imgIds=[184324])
anno = coco.loadAnns(_anno)
for i in range(len(anno)):
    print(anno[i]["bbox"])




