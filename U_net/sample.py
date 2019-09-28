from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

data_dir = r'D:\datasets\data'
label_dir = r'D:\datasets\label'


class MydataSet(Dataset):
    def __init__(self):
        super(MydataSet, self).__init__()
        self.dataset = os.listdir(data_dir)
        self.dataset.sort(key=lambda x: int(x.split('.')[0]))

    def __getitem__(self, index):
        image = Image.open(os.path.join(data_dir, self.dataset[index]))
        label = Image.open(os.path.join(label_dir, self.dataset[index]))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_data = transform(image)
        lable_data = transform(label)
        return image_data, lable_data

    def __len__(self):
        return len(self.dataset)
