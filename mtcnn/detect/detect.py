"""face detection and alignment using multi-task cascaded convolutional network. """

import torch
import torchvision.transforms as transforms
from test import utils


class Detector:
    """Joint face detection and alignment using MTCNN. """

    def __init__(self, pnet_model, rnet_model, onet_model):
        self.PNet = pnet_model
        self.RNet = rnet_model
        self.ONet = onet_model
        self.__device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.__transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.PNet.to(self.__device)
        self.RNet.to(self.__device)
        self.ONet.to(self.__device)

        self.PNet.eval()
        self.RNet.eval()
        self.ONet.eval()

    def detect(self, image):
        """Using unified cascaded CNNs to do face detection and alignment.
           The proposed CNNs consist of three stages:
           a. produces candidate windows quickly through a shallow CNN (Proposal Network).
           b. refines the windows by rejecting a large number of non-faces windows through a more
              complex CNN (Refinement Network).
           c. uses a more powerful CNN (Output Network) to refine the result again and output
              five facial landmarks position.
        """
        _cons, _boxes, _landmarks = self.__detect_pnet(image)
        if _boxes.size(0) != 0:
            _cons, _boxes, _landmarks = self.__detect_rnet(image, _boxes)
            if _boxes.size(0) != 0:
                _cons, _boxes, _landmarks = self.__detect_onet(image, _boxes)

        return _cons, _boxes, _landmarks

    def __detect_pnet(self, image, zoom=0.852, thres_c=0.886, thres_i=0.27):
        """Produces candidate windows quickly. """
        w, h = image.size
        min_side_len = min(w, h)

        scale = 1
        confidences, bboxes = torch.tensor([]), torch.tensor([])
        loffset, boffset = torch.tensor([]), torch.tensor([])
        while min_side_len >= 12:
            img_data = self.__transform(image).to(self.__device)
            img_data.unsqueeze_(0)

            _cls, _offset, _landmark = self.PNet(img_data)

            cls, offset, landmark = _cls[0, 0].cpu().data, _offset[0].cpu().data, _landmark[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, thres_c))
            if idxs.size(0) != 0:
                confidences = torch.cat([confidences, cls[idxs[:, 0], idxs[:, 1]]])
                bboxes = torch.cat([bboxes, utils.pyramid_feature_mapping(idxs, scale)])
                loffset = torch.cat([loffset, landmark.permute((1, 2, 0))[idxs[:, 0], idxs[:, 1], :]])
                boffset = torch.cat([boffset, offset.permute((1, 2, 0))[idxs[:, 0], idxs[:, 1], :]])

            scale *= zoom
            _w, _h = int(scale * w), int(scale * h)
            image = image.resize((_w, _h))
            min_side_len = min(_w, _h)

        boxes = utils.cord_regression(bboxes, boffset)
        landmarks = utils.cord_regression(bboxes, loffset)
        keep = utils.nms(boxes, confidences, threshold=thres_i)
        return confidences[keep], boxes[keep], landmarks[keep]

    def __detect_rnet(self, image, cboxes):
        """Refines the windows by rejecting a large number non-faces windows. """
        return self.__detect_model(self.RNet, image, cboxes, 0.9, 0.36, size=24)

    def __detect_onet(self, image, cboxes):
        """Refines the result again and outputs five landmark positions. """
        return self.__detect_model(self.ONet, image, cboxes, 0.999999, 0.49, size=48, nms_mode='minimum')

    def __detect_model(self, model, image, cboxes, thres_c, thres_i, size, nms_mode='union'):
        """detection of Refinement Network and Output Network. """
        bboxes = utils.convert_to_square(cboxes)
        img_data = self.__crop_boxes(image, bboxes, size)
        img_data = img_data.to(self.__device)

        _cls, _offset, _landmark = model(img_data)

        cls, offset, landmark = _cls[:, 0].cpu().data, _offset.cpu().data, _landmark.cpu().data
        mask = torch.gt(cls, thres_c)

        confidences, loffset, boffset = cls[mask], landmark[mask], offset[mask]
        if not any(mask):
            return confidences, loffset, boffset

        boxes = utils.cord_regression(bboxes[mask], boffset)
        landmarks = utils.cord_regression(bboxes[mask], loffset)
        keep = utils.nms(boxes, confidences, threshold=thres_i, mode=nms_mode)

        return confidences[keep], boxes[keep], landmarks[keep]

    def __crop_boxes(self, image, boxes, size):
        img_data = []
        for _box in boxes.tolist():
            _img = image.crop(_box)
            _img = _img.resize((size, size))
            _img_data = self.__transform(_img)
            img_data.append(_img_data)

        return torch.stack(img_data)


if __name__ == '__main__':
    from test.cfg import MODEL_SAVED_DIR, MODEL, TEST_DIR
    from test.info import Info
    import os
    from PIL import Image
    import cv2
    import numpy as np


    def _model(key):
        checkpoint = torch.load(MODEL_SAVED_DIR[key])
        model = MODEL[key]()
        model.load_state_dict(checkpoint)
        return model


    def _show(image, confs, boxes, landmark):
        _img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        print(_img)
        for _conf, _box, _landmark in zip(confs.numpy(), boxes.numpy(), landmark.numpy()):
            cv2.rectangle(_img, tuple(_box[:2]), tuple(_box[2:]), (255, 255, 0))
            _r = int(np.ceil((_box[2] - _box[0]) / 256))
            cv2.putText(_img, f"{_conf:.3f}", (_box[0], _box[1] - 2), 6, 0.3 * _r, (200, 75, 200))
            for xy in _landmark.reshape(-1, 2):
                cv2.circle(_img, tuple(xy), _r, (200, 110, 255), 2 * _r)

        cv2.imshow("mtcnn", _img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return _img


    def show_video(video):
        cap = cv2.VideoCapture(video)
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        while cap.isOpened():
            ret, img = cap.read()
            if ret == True:
                detector = Detector(_model("pnet"), _model("rnet"), _model("onet"))
                new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                confs, boxes, landmark = detector.detect(new_img)
                print(boxes)
                print("boxes: ", len(boxes))
                for _conf, _box, _landmark in zip(confs.numpy(), boxes.numpy(), landmark.numpy()):
                    cv2.rectangle(img, tuple(_box[:2]), tuple(_box[2:]), (255, 255, 0))
                    _r = int(np.ceil((_box[2] - _box[0]) / 256))
                    cv2.putText(img, f"{_conf:.3f}", (_box[0], _box[1] - 2), 6, 0.3 * _r, (200, 75, 200))
                    for xy in _landmark.reshape(-1, 2):
                        cv2.circle(img, tuple(xy), _r, (200, 110, 255), 2 * _r)
                cv2.imshow('Frame', img)
                if cv2.waitKey(16) & 0xFF == ord('q'):
                    break
            else:
                break
            cap.release()
            cv2.destroyAllWindows()


    cap = cv2.VideoCapture("/home/yzs/video/1.wmv")
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, img = cap.read()
        if ret == True:
            detector = Detector(_model("pnet"), _model("rnet"), _model("onet"))
            new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            confs, boxes, landmark = detector.detect(new_img)
            print(boxes)
            print("boxes: ", len(boxes))
            for _conf, _box, _landmark in zip(confs.numpy(), boxes.numpy(), landmark.numpy()):
                cv2.rectangle(img, tuple(_box[:2]), tuple(_box[2:]), (255, 255, 0))
                _r = int(np.ceil((_box[2] - _box[0]) / 256))
                cv2.putText(img, f"{_conf:.3f}", (_box[0], _box[1] - 2), 6, 0.3 * _r, (200, 75, 200))
                for xy in _landmark.reshape(-1, 2):
                    cv2.circle(img, tuple(xy), _r, (200, 110, 255), 2 * _r)
            cv2.imshow('Frame', img)
            if cv2.waitKey(16) & 0xFF == ord('q'):
                break
        else:
            break
        cap.release()
        cv2.destroyAllWindows()

    # detector = Detector(_model("pnet"), _model("rnet"), _model("onet"))
    # img_dir = TEST_DIR
    # img_pths = os.listdir(img_dir)
    # img = Image.open(os.path.join(img_dir, "2.jpeg"))
    # cons, boxes, landmark = detector.detect(img)
    # print(len(boxes))
    # if boxes.size(0) == 0:
    #     print("No faces detected!")
    # else:
    #     _img = _show(img, cons, boxes, landmark)

    # for _ipth in img_pths:
    #     img = Image.open(os.path.join(img_dir, _ipth))
    #     cons, boxes, landmark = detector.detect(img)
    #     print(len(boxes))
    #     if boxes.size(0) == 0:
    #         print("No faces detected!")
    #     else:
    #         _img = _show(img, cons, boxes, landmark)
