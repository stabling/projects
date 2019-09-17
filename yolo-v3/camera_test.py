import torchvision
from detector import *
import time
import cv2
from PIL import Image

if __name__ == '__main__':
    detector = Detector()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            frames = frame[:, :, ::-1]

            image = Image.fromarray(frames, 'RGB')
            width, high = image.size
            x_w = width / 416
            y_h = high / 416
            cropimg = image.resize((416, 416))
            imgdata = transforms(cropimg)
            imgdata = torch.FloatTensor(imgdata).view(-1, 3, 416, 416).cuda()

            y = detector(imgdata, 0.5, cfg.ANCHORS_GROUP)[0]

            for i in y:
                x1 = int((i[0]) * x_w)
                y1 = int((i[1]) * y_h)
                x2 = int((i[2]) * x_w)
                y2 = int((i[3]) * y_h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

            end_time = time.time()
            print(end_time - start_time)
        cv2.imshow('a', frame)
        cv2.waitKey(10)
