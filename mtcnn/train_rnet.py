from nets import BaseNet
import trainer

if __name__ == '__main__':
    r_net = BaseNet.RNet()

    train = trainer.Trainer(r_net, "rnet", "/home/yzs/image_data", "./model/r_net.pt")
    train.trainer()
