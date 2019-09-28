import torch
from torch import nn
from Unet_ import Unet
from sample import MydataSet
from torch.utils.data import DataLoader
from torch import optim
from evaluation import get_DC

if __name__ == '__main__':
    dataset = MydataSet()
    dataloader = DataLoader(dataset,batch_size=3,shuffle=True,num_workers=4)
    module = Unet(1,1).cuda()
    optimizer = optim.Adam(module.parameters())
    criticizer = nn.BCELoss()
    for epoch in range(1000):
        for data,lable in dataloader:
            data = data.cuda()
            lable = lable.cuda()
            output = module(data,L=2)
            loss1 = criticizer(output,lable)
            # loss2 = 1 - get_DC(output.cpu(),lable.cpu())
            # print(loss1.item(), loss2.item())
            # loss = loss1 + loss2
            # print(loss1.item(),loss2)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            print(loss1.item())
        torch.save(module.state_dict(),r'params\params{}.pt'.format(epoch))