import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from torchvision import transforms

from utils import nms, convert_to_square, _box
from nets import BaseNet
import cv2


class Detector:

    def __init__(self, pnet_param="./model/p_net.pt", rnet_param="./model/r_net.pt",
                 onet_param="./model/o_net.pt"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pnet = BaseNet.PNet().to(self.device)
        self.rnet = BaseNet.RNet().to(self.device)
        self.onet = BaseNet.ONet().to(self.device)

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):
        p_start_time = time.time()
        pnet_boxes = self.__detect_pnet(image)
        print("p_len: ", len(pnet_boxes))
        if len(pnet_boxes) == 0:
            return np.array([])
        p_end_time = time.time()
        pnet_time = p_end_time - p_start_time
        # print("pnet_time: ", pnet_time)

        r_start_time = time.time()
        rnet_boxes = self.__detect_rnet(image, pnet_boxes)
        print("r_len: ", len(rnet_boxes))
        if len(rnet_boxes) == 0:
            return np.array([])
        r_end_time = time.time()
        rnet_time = r_end_time - r_start_time

        o_start_time = time.time()
        onet_boxes = self.__detect_onet(image, rnet_boxes)
        print("o_len: ", len(onet_boxes))
        if len(onet_boxes) == 0:
            return np.array([])
        o_end_time = time.time()
        onet_time = o_end_time - o_start_time
        # print("onet_time: ", onet_time)
        sum_time = pnet_time + rnet_time + onet_time

        print("time: ", sum_time, " p_time: ", pnet_time, " r_time: ", rnet_time, " o_time: ", onet_time)

        return onet_boxes

    def __detect_pnet(self, image):

        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)
        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            img_data.unsqueeze_(0)
            img_data = img_data.to(self.device)
            _cls, _offset, _landmark = self.pnet(img_data)
            cls, offset, landmark = _cls[0][0].cpu().data, _offset[0].cpu().data, _landmark[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.6))
            boxes.extend(_box(idxs, offset, landmark, cls[idxs[:, 0], idxs[:, 1]], scale))

            if len(boxes) == 0:
                return np.array([])

            scale *= 0.7
            _w, _h = int(w * scale), int(h * scale)
            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return nms(np.stack(boxes), 0.6)

    def __detect_rnet(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = convert_to_square(pnet_boxes, image)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop([_x1, _y1, _x2, _y2])
            img = img.resize((24, 24))

            img_data = self.__image_transform(img).to(self.device)
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)
        _cls, _offset, _landmark = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        landmark = _landmark.cpu().data.numpy()

        index, _ = np.where(cls > 0.9)
        if len(index) == 0:
            return np.array([])
        boxes_ = pnet_boxes[index]
        cls = cls[index].reshape(-1, 1)
        offset = offset[index]
        landmark = landmark[index]
        wh = np.array([boxes_[:, i + 3] - boxes_[:, i + 1] for i in range(2)]).transpose(1, 0)
        print(np.tile(wh, (1, 2)))
        center = np.array([0.5 * (boxes_[:, i + 1] + boxes_[:, i + 3]) for i in range(2)]).transpose(1, 0)
        print(np.tile(center, (1, 2)))
        offset_boxes = offset * np.tile(wh, (1, 2)) + np.tile(center, (1, 2))
        landmark_boxes = landmark * np.tile(wh, (1, 5)) + np.tile(center, (1, 5))

        boxes = np.concatenate([cls, offset_boxes, landmark_boxes], axis=1)

        return nms(boxes, 0.6)

    def __detect_onet(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = convert_to_square(rnet_boxes, image)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop([_x1, _y1, _x2, _y2])
            img = img.resize((48, 48))

            img_data = self.__image_transform(img).to(self.device)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        _cls, _offset, _landmark = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        landmark = _landmark.cpu().data.numpy()

        index, _ = np.where(cls > 0.999)
        if len(index) == 0:
            return np.array([])
        boxes_ = rnet_boxes[index]
        cls = cls[index].reshape(-1, 1)
        offset = offset[index]
        landmark = landmark[index]
        wh = np.array([boxes_[:, i + 3] - boxes_[:, i + 1] for i in range(2)]).transpose(1, 0)
        center = np.array([0.5 * (boxes_[:, i + 1] + boxes_[:, i + 3]) for i in range(2)]).transpose(1, 0)
        offset_boxes = offset * np.tile(wh, (1, 2)) + np.tile(center, (1, 2))
        landmark_boxes = landmark * np.tile(wh, (1, 5)) + np.tile(center, (1, 5))
        boxes = np.concatenate([cls, offset_boxes, landmark_boxes], axis=1)

        return nms(boxes, 0.99, True)


if __name__ == '__main__':

    # img = Image.open("/home/yzs/pic/14.jpg")
    # font = ImageFont.truetype("DroidSerif-Italic.ttf", size=12)
    # detector = Detector()
    # boxes = detector.detect(img)
    # draw = ImageDraw.Draw(img)
    # for box in boxes:
    #     x_min, y_min, x_max, y_max = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    #     k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = int(box[5]), int(box[6]), int(box[7]), int(box[8]), int(box[9]), int(
    #             box[10]), int(box[11]), int(box[12]), int(box[13]), int(box[14])
    #     draw.rectangle((x_min, y_min, x_max, y_max))
    #     points_list = [(k1, k2), (k3, k4), (k5, k6), (k7, k8), (k9, k10)]
    #     for point in points_list:
    #         draw.point(point, fill=(0, 0, 255))
    # img.show()

    # opencv显示图片
    # img = cv2.imread("/home/yzs/pic/1.jpg")
    # font = ImageFont.truetype("DroidSerif-Italic.ttf", size=12)
    # detector = Detector()
    # new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # boxes = detector.detect(new_img)
    # for box in boxes:
    #     x_min, y_min, x_max, y_max = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    #     k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = int(box[5]), int(box[6]), int(box[7]), int(box[8]), int(box[9]), int(
    #         box[10]), int(box[11]), int(box[12]), int(box[13]), int(box[14])
    #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=1)
    #     cv2.putText(img, "cls: {}".format(box[0]), (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 0.8, color=(0, 255, 0))
    #     points_list = [(k1, k2), (k3, k4), (k5, k6), (k7, k8), (k9, k10)]
    #     for point in points_list:
    #         cv2.circle(img, point, 1, color=(0, 0, 255), thickness=1)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture('/home/yzs/video/1.wmv')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            detector = Detector()
            new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = detector.detect(new_img)
            i = 0
            for box in boxes:
                x_min, y_min, x_max, y_max = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = int(box[5]), int(box[6]), int(box[7]), int(box[8]), int(
                    box[9]), int(
                    box[10]), int(box[11]), int(box[12]), int(box[13]), int(box[14])
                if i % 3 == 0:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=1)
                    cv2.putText(img, "cls: {}".format(box[0]), (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 0.8,
                                color=(0, 255, 0))
                    points_list = [(k1, k2), (k3, k4), (k5, k6), (k7, k8), (k9, k10)]
                    for point in points_list:
                        cv2.circle(img, point, 1, color=(0, 0, 255), thickness=1)
                    i += 1
            cv2.imshow('Frame', img)

            if cv2.waitKey(16) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
