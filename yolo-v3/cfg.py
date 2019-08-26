IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

CLASS_NUM = 15

ANCHOR_GROUPS = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHOR_GROUPS_AREA = {
    13: [x * y for x, y in ANCHOR_GROUPS[13]],
    26: [x * y for x, y in ANCHOR_GROUPS[26]],
    52: [x * y for x, y in ANCHOR_GROUPS[52]],
}

CATEGORY_NAME = {
    0: "person",
    1: "car",
    2: "cat",
    3: "book",
    4: "computer",
    5: "lamp",
    6: "seat",
    7: "sofa",
    8: "plant",
    9: "bicycle",
    10: "motor",
    11: "horse",
    12: "light",
    13: "tv",
    14: "window"
}

