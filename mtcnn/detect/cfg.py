"""Configs for models training, saving, and loading.
    - MODEL_DIR/saved: models saving and loading directory
    - MODEL_DIR/model.info: information for models saved
    - MODEL_DIR/train.log: model training log
"""

import os

from nets.BaseNet import PNet, RNet, ONet

CFG_DIR = "/home/yzs/myproject/mtcnn/model"

MODEL_SAVED_DIR = {
    "pnet": os.path.join(CFG_DIR, "p_net.pt"),
    "rnet": os.path.join(CFG_DIR, "r_net.pt"),
    "onet": os.path.join(CFG_DIR, "o_net.pt"),
}

# TEST_DIR = "/home/yzs/video"
TEST_DIR = "/home/yzs/pic"

MODEL = {"pnet": PNet, "rnet": RNet, "onet": ONet}

BATCH_SIZE = 1024
LR = 0.001

# LOSS_FN = {"pnet": LossFn(1, 0.5, 0.5),
#            "rnet": LossFn(1, 0.5, 0.5),
#            "onet": LossFn(1, 0.5, 1),
#            }
