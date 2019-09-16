from nets import BaseNet
import trainer

if __name__ == '__main__':
    o_net = BaseNet.ONet()

    train = trainer.Trainer(o_net, "onet", "/home/yzs/image_data", "./model/o_net.pt")
    train.trainer()
