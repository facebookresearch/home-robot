import cv2
from pathlib import Path
GOALS = {
    # Object goals
    # "place_object_bed": {"type": "objectnav", "target": "bed",'action': 'place'},
    # "pick_object_bear": {"type": "objectnav", "target": "teddy bear",'action': "pick"},
    "object_chair": {"type": "objectnav", "target": "chair"},
    "object_couch": {"type": "objectnav", "target": "couch"},
    "object_plant": {"type": "objectnav", "target": "potted plant"},
    "object_bed": {"type": "objectnav", "target": "bed"},
    "object_toilet": {"type": "objectnav", "target": "toilet"},
    # "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
    "object_bear": {"type": "objectnav", "target": "teddy bear"},
    "object_oven": {"type": "objectnav", "target": "oven"},
    "object_sink": {"type": "objectnav", "target": "sink"},
    "object_refrigerator": {"type": "objectnav", "target": "refrigerator"},
    # "object_book": {"type": "objectnav", "target": "book"},
    # "object_person": {"type": "objectnav", "target": "person"},

    # Image goals
    "image_bear1": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/bear1.jpg"
        ),
    },
    "image_bed1": {
        "type": "imagenav",
        "target": "bed",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/bed1.jpg"
        ),
    },
    "image_bed2": {
        "type": "imagenav",
        "target": "bed",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/bed2.jpg"
        ),
    },
    "image_chair1": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair1.jpg"
        ),
    },
    "image_chair2": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair2.jpg"
        ),
    },
    "image_chair3": {
        "type": "imagenav",
        "target": "chair",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair3.jpg"
        ),
    },
    # "image_chair4": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair4.jpg"
        # ),
    # },
    # "image_chair5": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair5.jpg"
        # ),
    # },
    # "image_chair6": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/chair6.jpg"
        # ),
    # },
    "image_couch1": {
        "type": "imagenav",
        "target": "couch",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/couch1.jpg"
        ),
    },
    "image_couch2": {
        "type": "imagenav",
        "target": "couch",
        "image_path": f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/couch2.jpg",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/couch2.jpg"
        ),
    },
    "image_cup1": {
        "type": "imagenav",
        "target": "oven",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/cup1.jpg"
        ),
    },
    "image_oven1": {
        "type": "imagenav",
        "target": "oven",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/oven1.jpg"
        ),
    },
    "image_plant1": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/plant1.jpg"
        ),
    },
    "image_plant2": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/plant2.jpg"
        ),
    },
    "image_plant3": {
        "type": "imagenav",
        "target": "potted plant",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/plant3.jpg"
        ),
    },
    "image_refrigerator1": {
        "type": "imagenav",
        "target": "refrigerator",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/refrigerator1.jpg"
        ),
    },
    "image_sink1": {
        "type": "imagenav",
        "target": "sink",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/sink1.jpg"
        ),
    },
    "image_sink2": {
        "type": "imagenav",
        "target": "sink",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/sink2.jpg"
        ),
    },
    "image_toilet1": {
        "type": "imagenav",
        "target": "toilet",
        "image": cv2.imread(
            f"{str(Path(__file__).resolve().parent)}/airbnb1_goals/toilet1.jpg"
        ),
    },

    # Language goals
    "language_bear1": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The beige teddy bear.",
    },
    "language_bed1": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with the brown blanket and grey pillows.",
    },
    "language_bed2": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with the white blanket and white pillows",
    },
    "language_chair1": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The dark blue chair in the bedroom",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The grey and black chair in the kitchen",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The black office chair",
    },
    "language_couch1": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The black leather couch",
    },
    "language_couch2": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The couch covered in a white blanket",
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
        "instruction": "The small potted plant on the hallway table",
    },
    "language_plant2": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The large potted plant infront of the mirror",
    },
    "language_plant2": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The large potted plant infront of the window",
    },
    # "language_cup1": {
        # "type": "languagenav",
        # "target": "cup",
        # "landmarks": [],
        # "instruction": "The blue cup.",
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
    # "language_bottle1": {
        # "type": "languagenav",
        # "target": "bottle",
        # "landmarks": [],
        # "instruction": "Bottle of water.",
    # },
}
