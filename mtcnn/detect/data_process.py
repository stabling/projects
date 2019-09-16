"""Configs for data pre-processing. """

import os


SAMPLE_ID = {'negative': 0, 'positive': 1, 'part': 2, 'landmark': 3}

DATA_DIR = r"/home/data/MTCNN/image_data"
IMG_DIR = {
    "pnet": os.path.join(DATA_DIR, "pnet"),
    "rnet": os.path.join(DATA_DIR, "rnet"),
    "onet": os.path.join(DATA_DIR, "onet"),
}
ANNO_PTH = {
    "pnet_train": os.path.join(DATA_DIR, "Anno/pnet_train_annos.txt"),
    "rnet_train": os.path.join(DATA_DIR, "Anno/rnet_train_annos.txt"),
    "onet_train": os.path.join(DATA_DIR, "Anno/onet_train_annos.txt"),
    "pnet_val": os.path.join(DATA_DIR, "Anno/pnet_val_annos.txt"),
    "rnet_val": os.path.join(DATA_DIR, "Anno/rnet_val_annos.txt"),
    "onet_val": os.path.join(DATA_DIR, "Anno/onet_val_annos.txt"),
}