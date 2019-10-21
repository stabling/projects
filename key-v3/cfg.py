from torchvision import transforms

"""
------dataset参数------
IMAGE_WIDTH:图片的宽
IMAGE_HEIGHT：图片的高
CLASS_NUM：分类数
LABEL_FILE：标签文本路径
IMG_BASE_DIR：图片存放路径
TRANSFORM：图片转换格式
ANCHOR_GROUPS：锚框组
ANCHOR_GROUPS_AREA：各锚框的面积
CATEGORY_NAME：分类名
"""

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

CLASS_NUM = 5

LABEL_FILE = "/home/yzs/myproject/yolo-v3/Anno/label.txt"
IMG_BASE_DIR = "/home/yzs/myproject/yolo-v3/img"

TRANSFORM = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ANCHOR_GROUPS = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}
#
# ANCHOR_GROUPS_AREA = {
#     13: [x * y for x, y in ANCHOR_GROUPS[13]],
#     26: [x * y for x, y in ANCHOR_GROUPS[26]],
#     52: [x * y for x, y in ANCHOR_GROUPS[52]],
# }

# CATEGORY_NAME = {
#     0: "person",
#     1: "cat",
#     2: "computer",
#     3: "bicycle",
#     4: "elephant",
#     5: "horse",
#     6: "motor",
#     7: "window",
#     8: "lamp",
#     9: "sofa"
# }


"""
------trainer参数-----
TRAIN_BATCH_SIZE：训练的批次
VAL_BATCH_SIZE：验证的批次
IS_SHUFFLE：是否打乱
NUM_WORKERS：加载数据的线程
EPOCH：训练的批次
alpha：系数
"""

# TRAIN_BATCH_SIZE = 5
# VAL_BATCH_SIZE = 5
# IS_SHUFFLE = "True"
# NUM_WORKERS = 2
# EPOCH = 10000
# alpha = 0.9


"""
------detector参数-----
"""

TRAIN_BATCH_SIZE = 5
NUM_WORKERS = 2
EPOCH = 10000
alpha = 0.9
# CATEGORY_NAME = {
#     0: "person",
#     1: "cat",
#     2: "computer",
#     3: "bicycle",
#     4: "elephant",
#     5: "horse",
#     6: "motor",
#     7: "window",
#     8: "lamp",
#     9: "sofa"
# }

CATEGORY_NAME = {
    0: "ChenWeiTing",
    1: "ZhangBaiZhi",
    2: "XieTingFeng",
    3: "HuangZongZe",
    4: "LiuDeHua",
}
ANCHOR_GROUPS_AREA = {
    13: [x * y for x, y in ANCHOR_GROUPS[13]],
    26: [x * y for x, y in ANCHOR_GROUPS[26]],
    52: [x * y for x, y in ANCHOR_GROUPS[52]],
}


