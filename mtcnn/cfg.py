from torchvision import transforms

"""
------sample参数------
BASE_IMG_PATH:图像存放路径
LABEL_FILE：标签文本
LANDMARK_LABEL_FILE：关键点标签文本
SAVE_PATH：保存的路径
SAVE_LABEL_PATH：保存标签的路径
GEN_RATE：与图片生成的相关的比率
---------------
各参数含义： 正样本比率：  [中心点偏移量(标签宽高的倍率), 生成的边长(标签最小宽高的倍率), 生成的边长(标签最大宽高的倍率)]
          部分样本比率：  [中心点偏移量(标签宽高的倍率), 生成的边长(标签最小宽高的倍率), 生成的边长(标签最大宽高的倍率)]
          负样本比率：  [中心点(图片宽高的最小倍率), 中心点(图片宽高的最大倍率), 生成的边长(图片最小宽高的倍率)]
          关键点比率：  [中心点偏移量(标签宽高的倍率), 生成的边长(标签最小宽高的倍率), 生成的边长(标签最大宽高的倍率)]
---------------
FACE_SIZE： 定义三种不同的尺寸
NET_GROUP： 网络组
CONTROL_NUM_GROUP： 控制各种样本生成的数量
---------------
各参数含义： 正样本：  [遍历的次数, 丢弃的大小]
          部分样本：  [遍历的次数, 丢弃的大小]
          负样本：  [遍历的次数]
          关键点：  [遍历的次数, 丢弃的大小]
---------------
"""

BASE_IMG_PATH = "/home/yzs/celeba/img"
LABEL_FILE = "/home/yzs/celeba/anno/list_bbox.txt"
LANDMARK_LABEL_FILE = "/home/yzs/celeba/anno/list_landmarks.txt"
SAVE_PATH = "/home/yzs/save_face"
SAVE_LABEL_PATH = "/home/yzs/save_face/Anno"
GEN_RATE = {
    "POS_GEN_RATE": [0.1, 0.9, 1.2],
    "PART_GEN_RATE": [0.3, 0.6, 1.2],
    "NEG_GEN_RATE": [0.2, 0.8, 0.6],
    "LANDMARK_GEN_RATE": [0.03, 1, 1]
}
# 过拟合版本
# GEN_RATE = {
#     "POS_GEN_RATE": [0.1, 0.9, 1.2],
#     "PART_GEN_RATE": [0.3, 0.6, 1.2],
#     "NEG_GEN_RATE": [0.3, 0.9, 0.3],
#     "LANDMARK_GEN_RATE": [0.03, 1, 1]
# }
FACE_SIZE = [12, 24, 48]
NET_GROUP = {
    12: "pnet",
    24: "rnet",
    48: "onet"
}
EPOCH_NUM_GROUP = {
    "POS_EPOCH": 2,
    "PART_EPOCH": 2,
    "NEG_EPOCH": 6,
    "LANDMARK_EPOCH": 2
}
# 过拟合版本
# EPOCH_NUM_GROUP = {
#     "POS_EPOCH": 6,
#     "PART_EPOCH": 6,
#     "NEG_EPOCH": 40,
#     "LANDMARK_EPOCH": 6
# }


"""
------trainer参数------
BATCH_SIZE: 批次
EPOCHS： 训练的轮次
IS_SHUFFLE：是否打乱
NUM_WORKERS：加载数据集的线程
RATE_GROUP: 定义各网络比率组
THRES_GROUP: 判断召回率的阈值指标
"""

TRAIN_BATCH_SIZE = 1024
VAL_BATCH_SIZE = 2000
EPOCHS = 250
IS_SHUFFLE = True
NUM_WORKERS = 1
RATE_GROUP = {
    "pnet": [0.9, 0.5, 0.1],
    "rnet": [0.9, 0.9, 0.5],
    "onet": [0.5, 0.9, 0.9]
}
THRES_GROUP = {
    "pnet": [0.9, 0.1],
    "rnet": [0.9, 0.8],
    "onet": [0.999, 0.8]
}

"""
------detect参数------
SELECTIVE_PARAM： 
"""

SELECTIVE_PARAM = {
    "pnet": [0.9, 0.3],
    "rnet": [0.99, 0.1],
    "onet": [0.999, 0.3],
}

NAME_GROUP = ["pnet_log.txt", "rnet_log.txt", "onet_log.txt"]
