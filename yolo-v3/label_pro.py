from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir = r'E:\BaiduNetdiskDownload'
dataType = 'val2017'
annFile = r'{}\annotations\instances_{}.json'.format(dataDir, dataType)

valDir = r"E:\BaiduNetdiskDownload\val2017"
newDir = r"D:\project\project2.0\yolo-v3\coco_pro\new_pic"
# 初始化标注数据的 COCO api
coco = COCO(annFile)

# 获取一张图片的标签
for catIds in [2, 3, 4]:
    _anno = coco.getAnnIds(catIds=catIds)
    len_label = 12
    anno = coco.loadAnns(_anno)
    print(len(anno))
    for _anno in _anno:
        anno = coco.loadAnns(_anno)

        strs = (len_label - len(str(_anno))) * "0" + str(_anno) + ".jpg"
        img_path = os.path.join(valDir, strs)
        img = Image.open(img_path)
        img.save(os.path.join(newDir, "{}.jpg".format(i)))

        for i in range(len(anno)):
            b = []
            x1, y1, w, h = anno[i]["bbox"]
            i_id = anno[i]["image_id"]
            c_id = anno[i]["category_id"]
            label = np.array([c_id, x1, y1, w, h], dtype=np.int)
            b.append(label)

        c = np.stack(b)
        c = c[c[:, 0] < 5].reshape(-1)

        # 将数组读出来写到文件里
        with open(r"D:\project\project2.0\yolo-v3\label.txt", "a") as f:
            f.write(str(i_id) + ".jpg")
            f.write(" ")
            for i in c:
                f.write(str(i))
                f.write(" ")
            f.write("\n")
            f.flush()

f.close()
