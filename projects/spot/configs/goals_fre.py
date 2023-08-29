import cv2
from pathlib import Path
GOALS = {
    # Object goals
    "object_chair": {"type": "objectnav", "target": "chair"},
    "object_couch": {"type": "objectnav", "target": "couch"},
    "object_plant": {"type": "objectnav", "target": "potted plant"},
    "object_bed": {"type": "objectnav", "target": "bed"},
    "place_object_bed": {"type": "objectnav", "target": "bed",'action': 'place'},
    "object_toilet": {"type": "objectnav", "target": "toilet"},
    "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
    "object_bear": {"type": "objectnav", "target": "teddy bear"},
    "pick_object_bear": {"type": "objectnav", "target": "teddy bear",'action': "pick"},
    "object_oven": {"type": "objectnav", "target": "oven"},
    "object_sink": {"type": "objectnav", "target": "sink"},
    "object_refrigerator": {"type": "objectnav", "target": "refrigerator"},
    "object_book": {"type": "objectnav", "target": "book"},
    "object_person": {"type": "objectnav", "target": "person"},
    # Image goals
    "image_bed1": {
        "type": "imagenav",
        "target": "bed",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/bed1.png"
        ),
    },
    "image_chair1": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair1.png"
        ),
    },
    "image_chair2": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair2.png"
        ),
    },
    "image_chair3": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair3.png"
        ),
    },
    "image_chair4": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair4.png"
        ),
    },
    "image_chair5": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair5.png"
        ),
    },
    "image_chair6": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/chair6.png"
        ),
    },
    "image_couch1": {
        "type": "imagenav",
        "target": "couch",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/couch1.png"
        ),
    },
    "image_oven1": {
        "type": "imagenav",
        "target": "oven",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/oven1.png"
        ),
    },
    "image_plant1": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/plant1.png"
        ),
    },
    "image_plant2": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/plant2.png"
        ),
    },
    "image_plant3": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/plant3.png"
        ),
    },
    "image_refrigerator1": {
        "type": "imagenav",
        "target": "refrigerator",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/refrigerator1.png"
        ),
    },
    "image_sink1": {
        "type": "imagenav",
        "target": "sink",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/sink1.png"
        ),
    },
    "image_sink2": {
        "type": "imagenav",
        "target": "sink",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/sink2.png"
        ),
    },
    "image_toilet1": {
        "type": "imagenav",
        "target": "toilet",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/fremont_image_goals/toilet1.png"
        ),
    },
    # Language goals
    "language_bed1": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with white sheets.",
    },
    "language_chair1": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The brown leather chair next to the bedside table.",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The black plastic office chair.",
    },
    "language_chair3": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The brown leather chair next to the picture and plant.",
    },
    "language_chair4": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The grey armchair.",
    },
    "language_chair5": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The high chair with metal legs next to the kitchen counter.",
    },
    "language_chair6": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The chair with metal legs next to the dining table.",
    },
    "language_couch1": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        # "instruction": "The couch with colorful pillows.",
        "instruction": "The large white couch.",
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
        "instruction": "The green leafy plant next to the brown chair.",
    },
    "language_cup1": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The blue cup.",
    },
    "language_plant2": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The green leafy plant next to the grey armchair.",
    },
    "language_plant3": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The brown dead plant.",
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
    "language_bottle1": {
        "type": "languagenav",
        "target": "bottle",
        "landmarks": [],
        "instruction": "Bottle of water.",
    },
    "language_teddybear1": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The beige teddy bear.",
    },
}
