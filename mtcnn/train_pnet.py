from nets import BaseNet
import trainer

if __name__ == '__main__':

    p_net = BaseNet.PNet()

    train = trainer.Trainer(p_net, "pnet", "/home/yzs/image_data", "./model/pnet_59.pt")
    train.trainer()

