from PIL import Image
import os

path = r"/home/yzs/dataset/tiny_coco"
dataset = []
a = os.listdir(path)
dataset.extend(a)

for i, dataset in enumerate(dataset):
    img = Image.open(os.path.join(path, dataset))
    w, h = img.size
    print(w, h)
    _w = int(max(w, h))
    _h = int(max(w, h))
    new = Image.new(mode="RGB", size=(_w, _h), color=(255, 255, 255))
    x1 = int(0.5 * (_w - w))
    y1 = int(0.5 * (_h - h))
    new.paste(img, (x1, y1))
    print(new)
    resize_img = new.resize((416, 416))

    # img = img.resize((416, 416))
    resize_img.save(r"{}\{}".format(path, dataset))
