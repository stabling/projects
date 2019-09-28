from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import MainNet
import torch
import os
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from extend.draw import draw
from extend.lookahead import Lookahead
import torch.optim as optim
from extend.loss_fn import center_loss
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    device = torch.device("cuda")
    train_batch_size = 512
    val_batch_size = 512
    epoch = 10000
    load_path = "/home/yzs/PycharmProjects/face recognize/module/net.pt"
    net = MainNet().to(device)
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path))
    cls_criterion = nn.NLLLoss()
    optimizer = Lookahead(torch.optim.Adam(net.parameters()))
    writer = SummaryWriter()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.MNIST("datasets", train=False, download=False, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)

    for epoch in range(epoch):
        print("----------epoch{}-----------".format(epoch))
        for i, (input, target) in enumerate(train_dataloader):
            x, y = input.to(device), target.to(device)
            feature_out, cls_out = net(x, 1, 0.6)
            loss = cls_criterion(cls_out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("loss: ", loss.cpu().data.numpy())
            writer.add_scalar("loss", loss.cpu().data.numpy(), global_step=epoch)
            writer.add_histogram("weights", net.feature_layer[0].weight, global_step=epoch)

        with torch.no_grad():
            mean_rate = []
            for input, target in val_dataloader:
                x, y = input.to(device), target.to(device)
                feature_out, cls_out = net(x, 0.1, 0.1)
                correct_num = torch.sum((torch.argmax(cls_out, dim=1) == y))
                sum = len(y)
                rate = correct_num.float() / sum

                mean_rate.append(rate)

                draw(feature_out.cpu().data, cls_out.cpu().data)
            rate = torch.mean(torch.tensor(mean_rate))
            print("correct_rate is {}%".format(rate * 100))

        torch.save(net.state_dict(), "/home/yzs/PycharmProjects/face recognize/module/net.pt")
