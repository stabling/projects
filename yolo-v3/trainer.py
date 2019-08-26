import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from Darknet53 import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.4


def loss_func(output, target, alpha):
    output = output.permute(0, 2, 3, 1)
    output = output.view(output.size(0), output.size(1), output.size(2), 3, -1)
    obj = target[..., 0] > 0
    noobj = target[..., 0] == 0

    obj_loss = torch.mean((output[obj] - target[obj]) ** 2)
    noobj_loss = torch.mean((output[noobj] - target[noobj]) ** 2)

    loss = alpha * obj_loss + (1 - alpha) * noobj_loss

    return loss


if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(1000000):
        for i, (target_13, target_26, target_52, input) in enumerate(dataloader):
            target_13, target_26, target_52, input = target_13.to(device), target_26.to(device), target_52.to(
                device), input.to(device)
            output_13, output_26, output_52 = net(input)

            loss_13 = loss_func(output_13, target_13, 0.9)
            loss_26 = loss_func(output_26, target_26, 0.9)
            loss_52 = loss_func(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(" loss:", loss.item(), " loss_13:", loss_13.item(), " loss_26:", loss_26.item(), "loss_52",
                  loss_52.item())
            if epoch % 10 == 0:
                torch.save(net, "net.pth")
                print("epoch {} save success".format(epoch))

