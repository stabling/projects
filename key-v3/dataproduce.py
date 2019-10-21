import json
import os
from PIL import Image

label_path = "/home/yzs/backup/outputs"
img_path = "/home/yzs/backup/img"
out_file = "/home/yzs/myproject/yolo-v3/label_.txt"
file_name = os.listdir(label_path)
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
print(file_name)

with open(out_file, "w") as f:
    for file in file_name:
        cls = file.split("_")[0]
        img_name = file.replace("json", "jpg")
        image_path = os.path.join(img_path, img_name)
        image = Image.open(image_path)
        _w, _h = image.size
        path = os.path.join(label_path, file)
        label_txt = json.load(open(path))
        coordinate = label_txt["outputs"]["object"][0]["bndbox"]
        xmin, ymin, xmax, ymax = coordinate["xmin"], coordinate["ymin"], coordinate["xmax"], coordinate["ymax"]
        cx_, cy_, w_, h_ = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), (xmax - xmin), (ymax - ymin)
        w_scale, h_scale = _w / IMAGE_WIDTH, _h / IMAGE_HEIGHT
        cx, cy, w, h = int(cx_ / w_scale), int(cy_ / h_scale), int(w_ / w_scale), int(h_ / h_scale)
        f.write(image_path + " " + cls + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + " ")
        f.write("\n")
