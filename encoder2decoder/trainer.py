from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as utils
import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataset import MyData
from Net import MainNet
from utils.optims import Lookahead
from torch.utils.tensorboard import SummaryWriter


# Define the Command line argument(定义命令行参数，快速运行)
def argparser():
    """
    default argparse, please customize it by yourself. (默认的参数，可以赋值初始化此项)
    """
    parser = argparse.ArgumentParser(description="base parameters for network training")
    parser.add_argument("-r", "--resume", type=bool, default=True, help="if specified starts from checkpoint")
    parser.add_argument("-e", "--epochs", type=int, default=512, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-n", "--ncpu", type=int, default=4, help="number of cpu threads used during batch generation")
    parser.add_argument("-p", "--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("-c", "--chkpt_dir", type=str, default="checkpoints/", help="directory saved checkpoints")
    parser.add_argument("-m", "--module_name", type=str, default="autoencoder.pkl", help="save model name")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="learning rate for gradient descent")
    return parser.parse_args()


# Defile train class(定义训练的类)
class Trainer:

    def __init__(self, model, args=argparser):
        # 命令行参数部分
        self.args = args()
        if not os.path.exists(self.args.chkpt_dir):
            os.mkdir(self.args.chkpt_dir)
        self.device = torch.device(["cpu", "cuda"][torch.cuda.is_available()])
        self.net = model().to(self.device)
        # 非命令行参数部分
        self.path = "/home/yzs/img"
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        self.train_dataset = MyData(self.path)
        self.val_dataset = MyData(self.path)
        self.train_loader = DataLoader(self.train_dataset, self.args.batch_size, shuffle=True,
                                       num_workers=self.args.ncpu)
        self.val_loader = DataLoader(self.val_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.ncpu)

        self.optimizer = Lookahead(
            torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate))
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter()
        self.epoch = 0

    def main(self):
        if self.args.resume:
            if os.path.exists(os.path.join(self.args.chkpt_dir, self.args.module_name)):
                print("loading model '{}' ...".format(self.args.module_name))
                self.net.load_state_dict(torch.load(os.path.join(self.args.chkpt_dir, self.args.module_name)))
                print("loading finish!")

        while self.epoch < self.args.epochs:
            print("epoch------------", self.epoch + 1)
            self.epoch += 1
            self.train()

    def train(self):
        for i, x in enumerate(self.train_loader):
            x = x.to(self.device)
            out = self.net(x)
            loss = self.loss_fn(out.reshape(-1, 32*32*3), x.reshape(-1, 32*32*3))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.args.print_freq == 0:
                print("loss: ", loss.item())
            if i == len(self.train_loader) - 1:
                show_x = x[:32].detach().cpu()
                show_out = out[:32].detach().cpu()
                show_pic = torch.cat([show_x, show_out], dim=0)
                plt.imshow(np.transpose(utils.make_grid(show_pic), (1, 2, 0)))
                plt.show()
            self.writer.add_scalar("loss", loss.detach().cpu().numpy(), global_step=i % self.args.print_freq)
        torch.save(self.net.state_dict(), os.path.join(self.args.chkpt_dir, self.args.module_name))

    def validate(self):
        mean_rate = []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.net(x)

                correct_num = torch.sum((torch.argmax(out, dim=1) == y))
                sum = len(y)
                rate = correct_num.float() / sum
                mean_rate.append(rate)
            rate = torch.mean(torch.tensor(mean_rate))
            print("correct_rate is {}%".format(rate * 100))


if __name__ == '__main__':
    train = Trainer(MainNet)
    train.main()
