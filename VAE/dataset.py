from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])


class MyData(Dataset):

    def __init__(self, root):
        self.transform = data_transform
        self.root = root
        self.list = []
        self.list.extend(os.listdir(root))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.list[index])
        img = Image.open(img_path)
        img_data = data_transform(img)
        return img_data

