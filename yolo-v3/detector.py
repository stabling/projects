import cfg
from darknet53 import *
from utils import *

device = torch.device(cfg.DEVICE)


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = MainNet(cfg.CLASS_NUM).to(device)
        self.net.load_state_dict(torch.load('weights/darknet53.pt'))
        self.net.eval()

    def forward(self, input, thresh, anchors):

        output_13, output_26, output_52 = self.net(input.to(device))

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        boxes_all = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        last_boxes = []
        for n in range(input.size(0)):
            n_boxes = []
            boxes_n = boxes_all[boxes_all[:, 6] == n]
            for cls in range(cfg.CLASS_NUM):
                boxes_c = boxes_n[boxes_n[:, 5] == cls]
                if boxes_c.size(0) > 0:
                    n_boxes.extend(nms(boxes_c, 0.3))
                else:
                    pass
            last_boxes.append(torch.stack(n_boxes))
        return last_boxes

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        output = output.cpu()

        torch.sigmoid_(output[..., 4])  # 置信度加sigmoid激活
        torch.sigmoid_(output[..., 0:2])  # 中心点加sigmoid激活

        # 在计算置信度损失的时候使用的sigmoid
        mask = output[..., 4] > thresh
        idxs = mask.nonzero()
        vecs = output[mask]
        # print(vecs[..., 4])
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if idxs.size(0) == 0:
            return torch.Tensor([])

        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框
        c = vecs[:, 4]  # 置信度

        cls = torch.argmax(vecs[:, 5:], dim=1)

        cy = (idxs[:, 1].float() + vecs[:, 1]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 0]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 2])
        h = anchors[a, 1] * torch.exp(vecs[:, 3])

        w0_5, h0_5 = w / 2, h / 2
        x1, y1, x2, y2 = cx - w0_5, cy - h0_5, cx + w0_5, cy + h0_5

        return torch.stack([x1, y1, x2, y2, c, cls.float(), n.float()], dim=1)
