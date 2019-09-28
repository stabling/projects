import torch
import numpy as np
from Unet import Unet
from matplotlib import image
import cv2
from evaluation import get_DC
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    net = Unet(1, 1).cuda()
    net.load_state_dict(torch.load(r'params\unt_params.pt'))
    x = image.imread(r'D:\datasets\data\7.png')
    img = x
    x = torch.Tensor(x)
    x = x.view(-1, 1, 512, 512)
    x = x.cuda()
    out = net(x)
    x1 = image.imread(r'D:\datasets\label\7.png')
    x1 = torch.Tensor(x1)
    out = torch.reshape(out.cpu(), [512, 512])
    print('dice系数：', get_DC(out.cpu(), x1))
    out = out.cpu().detach().numpy()
    test_img = np.reshape(out, [512, 512])
    mask_obj = test_img > 0.85
    mask_noobj = test_img < 0.85
    test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGBA)
    indexs = np.array(np.where(mask_obj == True))
    indexs = np.stack(indexs, axis=1)
    test_img[indexs] = [0, 0, 200, 100]
    test_img[mask_noobj] = [255, 255, 255, 25]
    mask = Image.fromarray(np.uint8(test_img))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img * 255
    img = Image.fromarray(np.uint8(img))
    mask = mask.convert('RGBA')
    b, g, r, a = mask.split()
    img.paste(mask, (0, 0, 512, 512), mask=a)
    img = np.array(img)
    cv2.imshow('a', img)
    cv2.waitKey(0)
