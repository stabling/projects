from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from PIL import Image


class FaceDataset(Dataset):

    def __init__(self, path, net_name, mode):
        self.path = path
        self.net_name = net_name
        self.dataset = []
        if mode == "train":
            self.dataset.extend(open(os.path.join(path, "Anno/{}_train_annos.txt".format(net_name))).readlines())
        if mode == "val":
            self.dataset.extend(open(os.path.join(path, "Anno/{}_val_annos.txt".format(net_name))).readlines())
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, self.net_name, strs[0])
        cond = torch.tensor([int(strs[1])])
        offset = torch.tensor([float(strs[i]) for i in range(2, 6)])
        landmark = torch.tensor([float(strs[j]) for j in range(6, 16)])
        img_data = self.transform(Image.open(img_path))

        return img_data, cond, offset, landmark

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset("/home/yzs/image_data", "pnet", "val")
    img_data, cond, offset, landmark = dataset[10]
    print(cond, offset, landmark)
