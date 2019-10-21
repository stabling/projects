import torch
from torch.utils.data import DataLoader
from dataset_ import MyDataset
from nets.Darknet53 import Net
import utils
import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
                            num_workers=cfg.NUM_WORKERS)

    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(cfg.EPOCH):
        for i, (target_13, target_26, target_52, input) in enumerate(dataloader):
            target_13, target_26, target_52, input = target_13.to(device), target_26.to(device), target_52.to(
                device), input.to(device)
            output_13, output_26, output_52 = net(input)

            loss_13 = utils.loss_func(output_13, target_13, cfg.alpha)
            loss_26 = utils.loss_func(output_26, target_26, cfg.alpha)
            loss_52 = utils.loss_func(output_52, target_52, cfg.alpha)

            loss = loss_13 + loss_26 + loss_52  # 定义损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(" loss:", loss.item(), " loss_13:", loss_13.item(), " loss_26:", loss_26.item(), "loss_52",
                  loss_52.item())
            if epoch % 10 == 0:
                torch.save(net, "net.pth")
                print("epoch {} save success".format(epoch))
