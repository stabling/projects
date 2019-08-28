import torch


class en_loss:

    def main(self, logsigma, miu):
        return torch.mean(- torch.log(logsigma ** 2) + (miu ** 2) + (logsigma ** 2) - 1) * 0.5
