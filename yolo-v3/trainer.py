import dataset
import darknet53
import torch.nn
import cfg

device = torch.device(cfg.DEVICE)

# 损失函数定义
conf_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义置信度损失函数
center_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义中心点损失函数
wh_loss_fn = torch.nn.MSELoss()  # 宽高损失
cls_loss_fn = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失


def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    target = target.to(device)

    mask_obj = target[..., 4] > 0
    output_obj, target_obj = output[mask_obj], target[mask_obj]

    loss_obj_conf = conf_loss_fn(output_obj[:, 4], target_obj[:, 4])
    loss_obj_center = center_loss_fn(output_obj[:, 0:2], target_obj[:, 0:2])
    loss_obj_wh = wh_loss_fn(output_obj[:, 2:4], target_obj[:, 2:4])
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5].long())
    loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls

    # 负样本的时候只需要计算置信度损失
    mask_noobj = target[..., 4] == 0
    output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:, 4], target_noobj[:, 4])

    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = darknet53.MainNet(cfg.CLASS_NUM).to(device)
    net.train()

    opt = torch.optim.Adam(net.parameters())

    for epoch in range(10000):
        for target_13, target_26, target_52, img_data in train_loader:
            output_13, output_26, output_52 = net(img_data.to(device))

            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss.item())
