import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from utils import save_file
from data_process.dataset import FaceDataset
import cfg
from extend.lookahead import Lookahead
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, net, net_name, path, save_path):
        """
        初始化训练器
        :param net: 使用的网络
        :param net_name: 网络的名称
        :param train_path: 训练集的地址
        :param train_path: 验证集的地址
        :param save_path: 模型保存的地址
        """
        self.net_name = net_name
        self.path = path
        self.save_path = save_path
        self.backup_path = "./model"
        self.device = torch.device("cuda")
        self.net = net.to(self.device)
        self.cls_loss_fn = nn.BCELoss()
        self.landmark_loss_fn = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.optimizer = Lookahead(torch.optim.Adam(self.net.parameters(), weight_decay=0.0001))
        self.filename = os.path.join("/home/yzs/myproject/mtcnn/log", "{}_log.txt".format(self.net_name))
        self.writer = SummaryWriter(log_dir=f'/home/yzs/myproject/mtcnn/runs/{self.net_name}')

        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))
        self.net.train()

    def trainer(self):
        train_dataset = FaceDataset(self.path, self.net_name, "train")
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=cfg.IS_SHUFFLE,
                                      num_workers=cfg.NUM_WORKERS, pin_memory=True)

        val_dataset = FaceDataset(self.path, self.net_name, "val")
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=cfg.IS_SHUFFLE,
                                    num_workers=cfg.NUM_WORKERS, pin_memory=True)
        for epoch in range(cfg.EPOCHS):
            print("-----epoch-----{}".format(epoch))
            for i, (x, cls, offset, landmark) in enumerate(train_dataloader):
                x, cls, offset, landmark = x.to(self.device), cls.to(self.device), offset.to(self.device), landmark.to(
                    self.device)
                _out_cls, _out_offset, _out_landmark = self.net(x)
                out_cls = _out_cls.view(-1, 1)
                out_offset = _out_offset.view(-1, 4)
                out_landmark = _out_landmark.view(-1, 10)

                mask_index = torch.lt(cls, 2)  # 挑选出分类小于2的标签做分类损失
                cls_select = cls[mask_index]
                out_cls = out_cls[mask_index]
                # 困难样本训练
                cls_loss = self.cls_loss_fn(out_cls, cls_select.float())
                # cls_loss = torch.mean(
                #     torch.sort(torch.mean(cls_loss, dim=1), descending=True)[0][:int(len(cls_loss) * 0.7)])

                cls_index = (cls > 0) & (cls < 3)  # 挑选出分类大于0小于3的标签做分类损失
                offset = offset[cls_index[:, 0]]
                out_offset = out_offset[cls_index[:, 0]]
                # 困难样本训练
                offset_loss = self.criterion(out_offset, offset.float())
                # offset_loss = torch.mean(
                #     torch.sort(torch.mean(offset_loss, dim=1), descending=True)[0][:int(len(offset_loss) * 0.7)])

                off_landmark = torch.gt(cls, 2)[:, 0]  # 挑选出分类大于2的标签做关键点损失
                landmark = landmark[off_landmark]
                out_landmark = out_landmark[off_landmark]
                # 困难样本训练
                landmark_loss = self.landmark_loss_fn(out_landmark, landmark.float())
                # landmark_loss = torch.mean(
                #     torch.sort(torch.mean(landmark_loss, dim=1), descending=True)[0][:int(len(landmark_loss) * 0.7)])

                loss = cfg.RATE_GROUP[self.net_name][0] * cls_loss + cfg.RATE_GROUP[self.net_name][1] * offset_loss + \
                       cfg.RATE_GROUP[self.net_name][2] * landmark_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("epoch: ", epoch, "net_name:", self.net_name,
                      "loss: {0},cls_loss: {1},offset_loss: {2},landmark_loss: {3}".
                      format(loss.cpu().detach().numpy(), cls_loss.cpu().detach().numpy(),
                             offset_loss.cpu().detach().numpy(), landmark_loss.cpu().detach().numpy()))
                save_file(self.filename,
                          ["train_mode---", str(epoch), "net_name:", str(self.net_name), "loss:",
                           str(loss.cpu().detach().numpy()), "cls_loss:", str(cls_loss.cpu().detach().numpy()),
                           "offset_loss:", str(offset_loss.cpu().detach().numpy()), "landmark_loss:",
                           str(landmark_loss.cpu().detach().numpy())])
                self.writer.add_scalar("loss", loss.cpu().detach().numpy(), global_step=i * 512)
                self.writer.add_histogram("weights", self.net.layer[0].weight, global_step=i * 512)

                if i % 50 == 0:
                    correct_index = torch.eq(cls, 1)
                    error_index = torch.eq(cls, 0)
                    correct_out_index = torch.gt(_out_cls[correct_index], cfg.THRES_GROUP[self.net_name][0])
                    error_out_index = torch.lt(_out_cls[error_index], cfg.THRES_GROUP[self.net_name][1])

                    TP = torch.sum(correct_out_index)
                    TN = torch.sum(error_out_index)
                    FP = torch.sum(correct_index) - TP
                    FN = torch.sum(error_index) - TN

                    Accuracy = (TP + TN).float() / (TP + FP + TN + FN)
                    Precision = TP.float() / (TP + FP)
                    Recall = TP.float() / (TP + FN)

                    PR_curve = 0.5 * (Precision + Recall)

                    print(
                        "准确度: {}%, 精确度: {}%, 召回率: {}%, 综合平均: {}%".format(Accuracy.cpu().numpy() * 100,
                                                                         Precision.cpu().numpy() * 100,
                                                                         Recall.cpu().numpy() * 100,
                                                                         PR_curve.cpu().numpy() * 100))
                    save_file(self.filename,
                              ["准确度:", str(Accuracy.cpu().numpy() * 100), "精确度:",
                               str(Precision.cpu().numpy() * 100), "召回率:", str(Recall.cpu().numpy() * 100), "综合平均:",
                               str(PR_curve.cpu().numpy() * 100)])

            torch.save(self.net.state_dict(), os.path.join(self.backup_path, "{}_{}.pt".format(self.net_name, epoch+21)))
            print("-----save success-----")

            with torch.no_grad():
                for x, cls, _, _ in val_dataloader:
                    x, cls = x.to(self.device), cls.to(self.device)

                    out_cls, out_offset, out_landmark = self.net(x)

                    correct_index = torch.eq(cls, 1)
                    error_index = torch.eq(cls, 0)
                    correct_out_index = torch.gt(out_cls[correct_index], cfg.THRES_GROUP[self.net_name][0])
                    error_out_index = torch.lt(out_cls[error_index], cfg.THRES_GROUP[self.net_name][1])

                    TP = torch.sum(correct_out_index)
                    TN = torch.sum(error_out_index)
                    FP = torch.sum(correct_index) - TP
                    FN = torch.sum(error_index) - TN

                    Accuracy = (TP + TN).float() / (TP + FP + TN + FN)
                    Precision = TP.float() / (TP + FP)
                    Recall = TP.float() / (TP + FN)

                    PR_curve = 0.5 * (Precision + Recall)

                    save_file(self.filename,
                              ["val_mode---", str(epoch), "net_name:", str(self.net_name), "准确度:",
                               str(Accuracy.cpu().numpy() * 100), "精确度:", str(Precision.cpu().numpy() * 100), "召回率:",
                               str(Recall.cpu().numpy() * 100), "综合平均:", str(PR_curve.cpu().numpy() * 100)])

                    print(
                        "准确度: {}%, 精确度: {}%, 召回率: {}%, 综合平均: {}%".format(Accuracy.cpu().numpy() * 100,
                                                                         Precision.cpu().numpy() * 100,
                                                                         Recall.cpu().numpy() * 100,
                                                                         PR_curve.cpu().numpy() * 100))

                    self.writer.add_scalar("准确度", Accuracy.cpu().numpy(), global_step=epoch)
                    self.writer.add_scalar("精确度", Precision.cpu().numpy(), global_step=epoch)
                    self.writer.add_scalar("召回率", Recall.cpu().numpy(), global_step=epoch)
                    self.writer.add_scalar("综合平均", PR_curve.cpu().numpy(), global_step=epoch)
