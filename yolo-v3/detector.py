import cfg
import torch
from PIL import Image, ImageDraw, ImageFont
from utils import catergory
from torchvision import transforms
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = torch.load("net.pth")

        self.net.eval().to(device)

    def forward(self, input, thresh, anchors):
        input = input.to(device)
        output_13, output_26, output_52 = self.net(input)
        anchors[13], anchors[26], anchors[52] = torch.Tensor(anchors[13]).to(device), torch.Tensor(anchors[26]).to(
            device), torch.Tensor(anchors[52]).to(device)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        idxs_13, vecs_13 = idxs_13.to(device), vecs_13.to(device)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        idxs_26, vecs_26 = idxs_26.to(device), vecs_26.to(device)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        idxs_52, vecs_52 = idxs_52.to(device), vecs_52.to(device)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        mask = output[..., 0] > thresh

        idxs = mask.nonzero()
        vecs = output[mask]

        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        a = idxs[:, 3]

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        if len(vecs[:, 5:]) == 0:
            category = torch.Tensor([]).to(device)
        else:
            category = torch.argmax(vecs[:, 5:], dim=1).float()

        return torch.stack([vecs[:, 0].float(), cx, cy, w, h, category], dim=1)


if __name__ == '__main__':

    img = Image.open("/home/yzs/dataset/tiny_coco/{}.jpg".format(random.randint(1, 10)))
    img_data = transform(img)
    img_data = img_data.unsqueeze(dim=0)
    detector = Detector()
    start_time = time.time()
    vec = detector(img_data, 0.4, cfg.ANCHOR_GROUPS)
    end_time = time.time()
    time = end_time - start_time
    vec = vec.data.cpu()
    x1 = vec[:, 1] - 0.5 * vec[:, 3]
    y1 = vec[:, 2] - 0.5 * vec[:, 4]
    x2 = x1 + vec[:, 3]
    y2 = y1 + vec[:, 4]
    nms_vec = torch.stack([x1, y1, x2, y2, vec[:, 0], vec[:, 5]], dim=1)
    out_vec = catergory(nms_vec, 0.5)
    print(out_vec)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DroidSerif-Italic.ttf", size=12)
    for i in range(len(out_vec)):
        draw.rectangle((out_vec[i][0], out_vec[i][1], out_vec[i][2], out_vec[i][3]), outline=(0, 0, 255))
        draw.text(xy=(out_vec[i][0], out_vec[i][1]), text=cfg.CATEGORY_NAME[int(out_vec[i][5])], fill=(0, 255, 0),
                  font=font)
    print(time)
    img.show()
