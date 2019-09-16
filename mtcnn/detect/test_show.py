import cv2
import os
import glob
from PIL import Image, ImageDraw


class Img_Video_Show:
    def __init__(self, detector, video_path, img_path):
        self.ret_num = 0
        self.detector = detector
        self.video_path = video_path
        self.img_path = img_path

    def rectangle(self, rect, w_ratio, h_ratio):
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
        w_half = (rect[2] - rect[0]) * w_ratio / 2
        h_half = (rect[3] - rect[1]) * h_ratio / 2
        xcentre = rect[0] + (rect[2] - rect[0]) / 2
        ycentre = rect[1] + (rect[3] - rect[1]) / 2
        xmin = int(xcentre - w_half)
        ymin = int(ycentre - h_half)
        xmax = int(xcentre + w_half)
        ymax = int(ycentre + h_half)
        return xmin, ymin, xmax, ymax

    def video_show(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                self.ret_num += 1
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    _, _, ONet = self.detector.detect(image)
                    if self.ret_num % 1 == 0:
                        for onet in ONet:
                            xmin, ymin, xmax, ymax = self.rectangle(onet, 0.7, 0.66)
                            landmark = [(int(onet[5]), int(onet[6])), (int(onet[7]), int(onet[8])),
                                        (int(onet[9]), int(onet[10])), (int(onet[11]), int(onet[12])),
                                        (int(onet[13]), int(onet[14]))]
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
                            for center in landmark:
                                cv2.circle(frame, center, radius=1, color=(0, 255, 0), thickness=1)
                            cv2.putText(frame, text=str(onet[4]), org=(xmin, ymin),
                                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(255, 0, 255),
                                        thickness=1, lineType=1)
                except RuntimeError:
                    print("RuntimeError")
                except OverflowError:
                    print("OverflowError")
                except:
                    print("No Face")
                cv2.imshow("frame", frame)
                if cv2.waitKey(84) == ord("q"):
                    break
            else:
                break
        cap.release()  # 释放视频对象
        cv2.destroyAllWindows()  # 释放窗口

    def PIL_image_show(self, img_file):
        with Image.open(img_file) as img:
            Image.MAX_IMAGE_PIXELS = None
            img = img.convert("RGB")
            PNet, RNet, ONet = self.detector.detect(img)
            # Net = [PNet, RNet, ONet]
            Net = [ONet]
            for i in range(1):
                image = img.copy()
                imgDraw = ImageDraw.Draw(image)
                for net in Net[i]:
                    boxes = net[0:4]
                    land_mark = net[5:].astype(int)
                    xmin, ymin, xmax, ymax = self.rectangle(boxes, 0.7, 0.66)

                    landmark = [(land_mark[0], land_mark[1]), (land_mark[2], land_mark[3]),
                                (land_mark[4], land_mark[5]), (land_mark[6], land_mark[7]),
                                (land_mark[8], land_mark[9])]
                    imgDraw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=1)
                    imgDraw.point(landmark, fill=(0, 255, 0))
                    imgDraw.text([xmin, ymin], str(net[4]), (255, 0, 255))
                image.show()

    def CV2_image_show(self, img_file):
        _img = cv2.imread(img_file)
        image = Image.fromarray(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
        PNet, RNet, ONet = self.detector.detect(image)
        # Net = [PNet, RNet, ONet]
        Net = [ONet]
        for i in range(1):
            img = _img.copy()
            for net in Net[i]:
                boxes = net[0:4].astype(int)
                land_mark = net[5:].astype(int)
                xmin, ymin, xmax, ymax = self.rectangle(boxes, 0.7, 0.66)
                landmark = [(land_mark[0], land_mark[1]), (land_mark[2], land_mark[3]), (land_mark[4], land_mark[5]),
                            (land_mark[6], land_mark[7]), (land_mark[8], land_mark[9])]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
                for center in landmark:
                    cv2.circle(img, center, radius=1, color=(0, 255, 0), thickness=1)
                cv2.putText(img, text=str(net[4]), org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.3, color=(255, 0, 255), thickness=1, lineType=1)
            cv2.imshow("net", img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

    def show_test(self, state):  # state   0:视频显示   1:PIL显示图片   2: OPencv2显示图片
        if state == 0:
            self.video_show()
        else:
            image_path = os.path.join(self.img_path, "{}.jpg")
            img_number = len(glob.glob(os.path.join(self.img_path, '*.jpg')))
            for i in range(img_number):
                img_file = image_path.format(i)
                print("--------------------------{}.jpg--------------------------".format(i))
                try:
                    if state == 1:
                        self.PIL_image_show(img_file)
                    elif state == 2:
                        self.CV2_image_show(img_file)
                except RuntimeError:
                    print("RuntimeError")
                except OverflowError:
                    print("OverflowError")
                except:
                    print("No Face")
