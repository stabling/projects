from header import *
from module import *
from dataset import *
import traceback


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Trainer:

    def __init__(self):
        self.gpt2 = GPT2()

        self.weight_file_bak = os.path.join("weights", "apt2_k_bak.pt")
        self.weight_file = os.path.join("weights", "apt2_k.pt")
        if os.path.exists(self.weight_file):
            self.net.load_state_dict(torch.load(self.weight_file))
        else:
            self.net.apply(weight_init)

        self.net = nn.DataParallel(self.gpt2, device_ids=[0, 2, 3])
        self.net.to(torch.device(cfg.device))

        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        myDataset = MyDataset(r"C:\tok")
        print(len(myDataset))
        dataloader = DataLoader(myDataset, batch_size=10, shuffle=True)
        for epoch in range(10000):
            sum_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(torch.device(cfg.device)), y.to(torch.device(cfg.device))
                p = torch.arange(0, x.shape[1])[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
                # print(p)
                _y = self.net(x, p).reshape(-1, cfg.vocab_num)
                y = y.reshape(-1)
                loss = F.cross_entropy(_y, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(epoch, i, loss.cpu().detach().item())
                sum_loss += loss.cpu().detach().item()
                if i % 500 == 0 and i > 0:
                    torch.save(self.net.state_dict(), self.weight_file)
                    torch.save(self.net.state_dict(), self.weight_file_bak)

            print(epoch, sum_loss / len(dataloader))
            torch.save(self.net.state_dict(), self.weight_file)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
