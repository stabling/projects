import torch
from PIL import Image
from nets.mobilenet import Net
from torchvision import transforms


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    net.load_state_dict(torch.load("./module/net.pt", map_location="cpu"))
    img = Image.open("/home/yzs/data/mnist/7.jpg")
    img = img.convert("L")

    net.eval()

    transform = transforms.Compose([
        transforms.Resize(56),
        transforms.ToTensor()
    ])

    img_data = transform(img).unsqueeze(0).to(device)
    _, out_ = net(img_data)
    print(out_)
    out = torch.argmax(out_)
    print("The prediction is {}".format(out))