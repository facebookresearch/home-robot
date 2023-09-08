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
    "object_cup": {"type": "objectnav", "target": "cup"},
    "object_bowl": {"type": "objectnav", "target": "bowl"},
    "object_couch": {"type": "objectnav", "target": "couch"},
    "object_plant": {"type": "objectnav", "target": "potted plant"},
    "object_bed": {"type": "objectnav", "target": "bed"},
    "object_toilet": {"type": "objectnav", "target": "toilet"},
    "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
    "object_bear": {"type": "objectnav", "target": "teddy bear"},
    "object_book": {"type": "objectnav", "target": "book"},
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
        "instruction": "The beige teddy bear.",
    },
    "language_bear2": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The stuffed lion toy",
    },
    "language_bear3": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The green dinosaur stuffed toy.",
    },
    "language_bed1": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bunk bed with stars on the blanket.",
    },
    "language_bed2": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with blue and white sheets.",
    },
    "language_book1": {
        "type": "languagenav",
        "target": "book",
        "landmarks": [],
        "instruction": "The green cover book on the coffee table.",
    },
    "language_book2": {
        "type": "languagenav",
        "target": "book",
        "landmarks": [],
        "instruction": "The book on the desk. It has a car on the cover.",
    },
    "language_bowl1": {
        "type": "languagenav",
        "target": "bowl",
        "landmarks": [],
        "instruction": "The light blue bowl on the coffee table.",
    },
    # "language_bowl2": {
        # "type": "languagenav",
        # "target": "bowl",
        # "landmarks": [],
        # "instruction": "The bowl on the wood desk",
    # },
    "language_chair1": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The metal chair with a wooden seat.",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The grey cloth chair near the table.",
    },
    "language_chair3": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The bar-height chair at the kitchen island.",
    },
    # "language_chair4": {
        # "type": "languagenav",
        # "target": "chair",
        # "landmarks": [],
        # "instruction": "The office chair at the desk.",
    # },
    "language_couch1": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The large grey couch.",
    },
    # "language_couch2": {
        # "type": "languagenav",
        # "target": "couch",
        # "landmarks": [],
        # "instruction": "The large grey couch in the sunroom.",
    # },
    "language_cup1": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The red plastic cup on the table.",
    },
    "language_cup2": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The green mug on the kitchen island.",
    },
    "language_cup3": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The light blue cup.",
    },
    # "language_cup4": {
        # "type": "languagenav",
        # "target": "cup",
        # "landmarks": [],
        # "instruction": "The green mug on the plastic chair",
    # },
    "language_oven1": {
        "type": "languagenav",
        "target": "oven",
        "landmarks": [],
        "instruction": "The oven.",
    },
    # "language_plant1": {
        # "type": "languagenav",
        # "target": "potted plant",
        # "landmarks": [],
        # "instruction": "The potted plant next to the bed",
    # },
    # "language_plant2": {
        # "type": "languagenav",
        # "target": "potted plant",
        # "landmarks": [],
        # "instruction": "The potted plant in the living room",
    # },
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
    # "language_sink2": {
        # "type": "languagenav",
        # "target": "sink",
        # "landmarks": [],
        # "instruction": "The kitchen sink.",
    # },
    "language_toilet1": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The toilet.",
    },
    # "language_tv1": {
        # "type": "languagenav",
        # "target": "sink",
        # "landmarks": [],
        # "instruction": "The television.",
    # },
}

replace = {'bear': 'teddy bear', 'table': 'dining table','plant':'potted plant'}
folder = f"{str(Path(__file__).resolve().parent)}/airbnb8_goals"
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
