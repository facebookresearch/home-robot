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
    "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
    "object_bear": {"type": "objectnav", "target": "teddy bear"},
    "object_oven": {"type": "objectnav", "target": "oven"},
    "object_sink": {"type": "objectnav", "target": "sink"},
    "object_cup": {"type": "objectnav", "target": "cup"},
    "object_bowl": {"type": "objectnav", "target": "bowl"},
    "object_refrigerator": {"type": "objectnav", "target": "refrigerator"},
    # "object_book": {"type": "objectnav", "target": "book"},
    # "object_person": {"type": "objectnav", "target": "person"},

    # Image goals
    # "image_bear1": {
        # "type": "languagenav",
        # "target": "teddy bear",
        # "landmarks": [],
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/bear1.jpg"
        # ),
    # },
    # "image_bed1": {
        # "type": "imagenav",
        # "target": "bed",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/bed1.jpg"
        # ),
    # },
    # "image_bed2": {
        # "type": "imagenav",
        # "target": "bed",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/bed2.jpg"
        # ),
    # },
    # "image_chair1": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair1.jpg"
        # ),
    # },
    # "image_chair2": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair2.jpg"
        # ),
    # },
    # "image_chair3": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair3.jpg"
        # ),
    # },
    # "image_chair4": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair4.jpg"
        # ),
    # },
    # "image_chair5": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair5.jpg"
        # ),
    # },
    # "image_chair6": {
        # "type": "imagenav",
        # "target": "chair",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/chair6.jpg"
        # ),
    # },
    # "image_couch1": {
        # "type": "imagenav",
        # "target": "couch",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/couch1.jpg"
        # ),
    # },
    # "image_couch2": {
        # "type": "imagenav",
        # "target": "couch",
        # "image_path": f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/couch2.jpg",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/couch2.jpg"
        # ),
    # },
    # "image_cup1": {
        # "type": "imagenav",
        # "target": "oven",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/cup1.jpg"
        # ),
    # },
    # "image_oven1": {
        # "type": "imagenav",
        # "target": "oven",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/oven1.jpg"
        # ),
    # },
    # "image_plant1": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant1.jpg"
        # ),
    # },
    # "image_plant2": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant2.jpg"
        # ),
    # },
    # "image_plant3": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant3.jpg"
        # ),
    # },
    # "image_plant4": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant4.jpg"
        # ),
    # },
    # "image_plant5": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant5.jpg"
        # ),
    # },
    # "image_plant6": {
        # "type": "imagenav",
        # "target": "potted plant",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/plant6.jpg"
        # ),
    # },
    # # "image_refrigerator1": {
        # # "type": "imagenav",
        # # "target": "refrigerator",
        # # "image": cv2.imread(
            # # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/refrigerator1.jpg"
        # # ),
    # # },
    # "image_sink1": {
        # "type": "imagenav",
        # "target": "sink",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/sink1.jpg"
        # ),
    # },
    # "image_sink2": {
        # "type": "imagenav",
        # "target": "sink",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/sink2.jpg"
        # ),
    # },
    # "image_toilet1": {
        # "type": "imagenav",
        # "target": "toilet",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/toilet1.jpg"
        # ),
    # },
    # "image_toilet2": {
        # "type": "imagenav",
        # "target": "toilet",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/toilet2.jpg"
        # ),
    # },
    # "image_tv1": {
        # "type": "imagenav",
        # "target": "toilet",
        # "image": cv2.imread(
            # f"{str(Path(__file__).resolve().parent)}/airbnb2_goals/tv1.jpg"
        # ),
    # },

    # Language goals
    "language_bear1": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The stuffed lion",
    },
    "language_bear2": {
        "type": "languagenav",
        "target": "teddy bear",
        "landmarks": [],
        "instruction": "The beige teddy bear",
    },
    "language_bed1": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with white blanket and grey pillows",
    },
    "language_bed2": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with the white blanket pulled back halfway and grey sheets",
    },
    "language_bed2": {
        "type": "languagenav",
        "target": "bed",
        "landmarks": [],
        "instruction": "The bed with the white blanket pulled back diagonally showing white sheets",
    },
    "language_bowl1": {
        "type": "languagenav",
        "target": "bowl",
        "landmarks": [],
        "instruction": "The light blue bowl on the living room coffee table",
    },
    "language_bowl2": {
        "type": "languagenav",
        "target": "bowl",
        "landmarks": [],
        "instruction": "The lavender bowl on the kitchen counter",
    },

    "language_chair1": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "Black leather chairs in the kitchen",
    },
    "language_chair2": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The white plastic chair at the desk",
    },
    "language_chair3": {
        "type": "languagenav",
        "target": "chair",
        "landmarks": [],
        "instruction": "The brown chair in the living room",
    },
    "language_couch1": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The small grey couch",
    },
    "language_couch2": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The light brown couch with blue pillows",
    },
    "language_couch3": {
        "type": "languagenav",
        "target": "couch",
        "landmarks": [],
        "instruction": "The larage grey couch in front of the yellow wall",
    },
    "language_cup1": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The red cup on top of the brown chair.",
    },
    "language_cup2": {
        "type": "languagenav",
        "target": "cup",
        "landmarks": [],
        "instruction": "The green cup on the kitchen counter.",
    },
    "language_oven1": {
        "type": "languagenav",
        "target": "oven",
        "landmarks": [],
        "instruction": "The oven.",
    },
    "language_refrigerator1": {
        "type": "languagenav",
        "target": "refrigerator",
        "landmarks": [],
        "instruction": "The refrigerator.",
    },
    "language_plant1": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The large potted plant next to the foosball table",
    },
    "language_plant2": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The large potted plant next to the front door",
    },
    "language_plant3": {
        "type": "languagenav",
        "target": "potted plant",
        "landmarks": [],
        "instruction": "The potted plant infront of the sliding glass door",
    },
    # "language_plant4": {
        # "type": "languagenav",
        # "target": "potted plant",
        # "landmarks": [],
        # "instruction": "The potted plant near the kitchen and tv",
    # },
    # "language_plant5": {
        # "type": "languagenav",
        # "target": "potted plant",
        # "landmarks": [],
        # "instruction": "The potted plant next to the refrigerator",
    # },

    # "language_plant6": {
        # "type": "languagenav",
        # "target": "potted plant",
        # "landmarks": [],
        # "instruction": "The potted plant near the front door",
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
        "instruction": "The bathroom sink with marble top.",
    },
    "language_sink2": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The kitchen sink",
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
        "instruction": "The television mounted on a white wall.",
    },
    "language_tv2": {
        "type": "languagenav",
        "target": "sink",
        "landmarks": [],
        "instruction": "The television mounted on a yellow wall.",
    },
    # "language_bottle1": {
        # "type": "languagenav",
        # "target": "bottle",
        # "landmarks": [],
        # "instruction": "Bottle of water.",
    # },
}
replace = {'bear': 'teddy bear', 'table': 'dining table','plant': 'potted plant'}
folder = f"{str(Path(__file__).resolve().parent)}/airbnb3_goals"
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
