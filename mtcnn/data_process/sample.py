import numpy as np
import os
from utils import iou, center_gen_pic, round_gen_pic
from PIL import Image
import time
import cfg


def gen_pic(style, mode="train"):
    face_size = cfg.FACE_SIZE[style]

    print("-----START TO GEN {} PICTURES-----".format(cfg.NET_GROUP[face_size]))

    pos_img_path = os.path.join(cfg.SAVE_PATH, cfg.NET_GROUP[face_size], "positive")
    part_img_path = os.path.join(cfg.SAVE_PATH, cfg.NET_GROUP[face_size], "part")
    negative_img_path = os.path.join(cfg.SAVE_PATH, cfg.NET_GROUP[face_size], "negative")
    landmark_img_path = os.path.join(cfg.SAVE_PATH, cfg.NET_GROUP[face_size], "landmark")
    for path in [pos_img_path, part_img_path, negative_img_path, landmark_img_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    save_anno_file = open(os.path.join(cfg.SAVE_LABEL_PATH, "{0}_{1}_annos.txt".format(cfg.NET_GROUP[face_size], mode)),
                          "a")

    start_time = time.time()

    off_box = []
    landmark_box = []
    for i, line in enumerate(open(cfg.LABEL_FILE)):
        if i < 2:
            continue
        strs = line.strip().split()
        off_box.append(np.array(strs))
    off_boxes = np.stack(off_box)

    for i, line in enumerate(open(cfg.LANDMARK_LABEL_FILE)):
        if i < 2:
            continue
        strs = line.strip().split()
        landmark_box.append(np.array(strs[1:]))
    landmark_boxes = np.stack(landmark_box)

    boxes = np.concatenate([off_boxes, landmark_boxes], axis=1)

    pos_count = 0
    part_count = 0
    neg_count = 0
    landmark_count = 0

    for i, strs in enumerate(boxes):
        with Image.open(os.path.join(cfg.BASE_IMG_PATH, strs[0])) as img:
            img_w, img_h = img.size
            box = np.array(strs[1:], dtype=float)
            if box[0] <= 0 or box[1] <= 0 or box[2] <= 0 or box[3] <= 0:
                continue
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]

            crop_boxes, boxes = center_gen_pic(box, cfg.EPOCH_NUM_GROUP["POS_EPOCH"], cfg.GEN_RATE["POS_GEN_RATE"])
            a = iou(box, crop_boxes)
            for j, crop_box in enumerate(crop_boxes):
                if crop_box[0] <= 0 or crop_box[1] <= 0 or crop_box[2] > img_w or crop_box[3] > img_h:
                    continue
                if a[j] > 0.72:
                    crop_img = img.crop(crop_box)
                    resize_img = crop_img.resize((face_size, face_size))
                    resize_img.save(os.path.join(pos_img_path, "{0}_{1}.jpg".format(mode, pos_count)))

                    save_anno_file.write(
                        "positive/{0}_{1}.jpg {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}\n".format(
                            mode, pos_count, 1, boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3], 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0))
                    save_anno_file.flush()

                    pos_count += 1

            crop_boxes, boxes = center_gen_pic(box, cfg.EPOCH_NUM_GROUP["PART_EPOCH"], cfg.GEN_RATE["PART_GEN_RATE"])
            a = iou(box, crop_boxes)
            for j, crop_box in enumerate(crop_boxes):
                if crop_box[0] <= 0 or crop_box[1] <= 0 or crop_box[2] > img_w or crop_box[3] > img_h:
                    continue
                if a[j] < 0.424:
                    crop_img = img.crop(crop_box)
                    resize_img = crop_img.resize((face_size, face_size))
                    resize_img.save(os.path.join(part_img_path, "{0}_{1}.jpg".format(mode, part_count)))

                    save_anno_file.write(
                        "part/{0}_{1}.jpg {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}\n".format(
                            mode, part_count, 2, boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3], 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0))
                    save_anno_file.flush()

                    part_count += 1

            boxes = round_gen_pic(img_w, img_h, face_size, cfg.EPOCH_NUM_GROUP["NEG_EPOCH"],
                                  cfg.GEN_RATE["NEG_GEN_RATE"])
            a = iou(box, boxes)
            for k, boxes_ in enumerate(boxes):
                if boxes_[0] <= 0 or boxes_[1] <= 0 or boxes_[2] > img_w or boxes_[3] > img_h:
                    continue
                if a[k] < 0.01:
                    crop_img = img.crop(boxes_)
                    resize_img = crop_img.resize((face_size, face_size))
                    resize_img.save(os.path.join(negative_img_path, "{0}_{1}.jpg".format(mode, neg_count)))

                    save_anno_file.write(
                        "negative/{0}_{1}.jpg {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}\n".format(
                            mode, neg_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                    save_anno_file.flush()

                    neg_count += 1

            crop_boxes, boxes = center_gen_pic(box, cfg.EPOCH_NUM_GROUP["LANDMARK_EPOCH"],
                                               cfg.GEN_RATE["LANDMARK_GEN_RATE"])
            a = iou(box, crop_boxes)
            for j, crop_box in enumerate(crop_boxes):
                if crop_box[0] <= 0 or crop_box[1] <= 0 or crop_box[2] > img_w or crop_box[3] > img_h:
                    continue
                if a[j] > 0.734:
                    crop_img = img.crop(crop_box)
                    resize_img = crop_img.resize((face_size, face_size))
                    resize_img.save(os.path.join(landmark_img_path, "{0}_{1}.jpg".format(mode, landmark_count)))

                    save_anno_file.write(
                        "landmark/{0}_{1}.jpg {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}\n".format(
                            mode, landmark_count, 3, 0, 0, 0, 0, boxes[j, 4], boxes[j, 5], boxes[j, 6],
                            boxes[j, 7],
                            boxes[j, 8], boxes[j, 9], boxes[j, 10], boxes[j, 11], boxes[j, 12], boxes[j, 13]))

                    save_anno_file.flush()

                    landmark_count += 1

    print("sum of positive_pic is {}".format(pos_count))
    print("sum of part_pic is {}".format(part_count))
    print("sum of neg_pic is {}".format(neg_count))
    print("sum of landmark_pic is {}".format(landmark_count))

    end_time = time.time()
    use_time = (end_time - start_time) / 3600
    print(use_time)

    save_anno_file.close()
