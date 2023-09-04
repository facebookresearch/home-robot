import cv2
from pathlib import Path
GOALS = {
    # Object goals
    "place_object_bed": {"type": "objectnav", "target": "bed",'action': 'place'},
    "place_object_chair": {"type": "objectnav", "target": "chair",'action': 'place'},
    "place_object_couch": {"type": "objectnav", "target": "couch",'action': 'place'},
    "pick_object_bear": {"type": "objectnav", "target": "teddy bear",'action': "pick"},
    "pick_object_bottle": {"type": "objectnav", "target": "bottle",'action': "pick"},
    "object_chair": {"type": "objectnav", "target": "chair"},
    "object_couch": {"type": "objectnav", "target": "couch"},
    "object_plant": {"type": "objectnav", "target": "potted plant"},
    "object_bed": {"type": "objectnav", "target": "bed"},
    "object_toilet": {"type": "objectnav", "target": "toilet"},
    "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
    "object_bear": {"type": "objectnav", "target": "teddy bear"},
    "object_oven": {"type": "objectnav", "target": "oven"},
    "object_sink": {"type": "objectnav", "target": "sink"},
    "object_refrigerator": {"type": "objectnav", "target": "refrigerator"},
    # "object_book": {"type": "objectnav", "target": "book"},
    # "object_person": {"type": "objectnav", "target": "person"},

    # Image goals

    # Language goals
    "language_bear1": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The stuffed lion toy",
    },
    "language_bed1": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with white blanket and black frame.",
    },
    "language_bed2": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with the stuffed animal on top.",
    },
    "language_bowl1": {
        "type": "languagenav",
        "target": "bowl",
        "landmarks": [],
        "instruction": "The light blue bowl on the kitchen island",
    },
    "language_chair1": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The brown chair near the plant with an orange pillow.",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The brown dining table chair",
    },
    "language_chair3": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The grey chair with a white pillow.",
    },
    "language_chair4": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The small wooden chair at the desk.",
    },
    "language_couch1": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The white rectangular couch with no pillows.",
    },
    "language_couch2": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The large grey living room couch with many pillows.",
    },
    "language_cup1": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The red cup on top coffee table.",
    },
    "language_oven1": {
        "type": "languagenav",
        "target": "oven",
        "landmarks": [],
        "instruction": "The oven.",
    },
    "language_plant1": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The group of plants in front of the curtain",
    },
    "language_plant2": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The potted plant on top of the cabinet",
    },
    "language_refrigerator1": {
        "type": "languagenav",
        "target": "refrigerator",
        "landmarks": [],
        "instruction": "The refrigerator.",
    },
    "language_sink1": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The bathroom sink.",
    },
    "language_sink2": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The kitchen sink.",
    },
    "language_toilet1": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The toilet.",
    },
    "language_tv1": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The television.",
    },
}
replace = {'bear': 'teddy bear', 'table': 'dining table'}
folder = f"{str(Path(__file__).resolve().parent)}/airbnb4_goals"
import os
for img in os.listdir(folder):
    if img.endswith(".jpg"):
        name = img.split(".")[0]
        obj_class = name[:-1]
        if obj_class in replace:
            obj_class = replace[obj_class]
        GOALS[f'image_{name}'] = {
            "type": "imagenav",
            "target": obj_class,
            "image": cv2.imread(f"{folder}/{img}"),
        }
