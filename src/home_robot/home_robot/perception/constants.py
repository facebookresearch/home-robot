# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)

hm3d_to_mp3d_path = Path(__file__).resolve().parent / "matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep="    ", header=0, engine="python")
hm3d_to_mp3d = {row["category"]: row["mpcat40index"] for _, row in df.iterrows()}
hm3d_raw_to_hm3d = {row["raw_category"]: row["category"] for _, row in df.iterrows()}
all_hm3d_categories = [row["category"] for _, row in df.iterrows()]


# Color constants we use.
# Note: originally from Habitat
# from habitat_sim.utils.common import d3_40_colors_rgb
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


class SemanticCategoryMapping(ABC):
    """
    This class contains a mapping from semantic and goal category IDs provided by
    a Habitat environment to category IDs stored in the semantic map, as well as
    the color palettes and legends to visualize these categories.
    """

    def __init__(self):
        pass

    @abstractmethod
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        pass

    @abstractmethod
    def reset_instance_id_to_category_id(self, env):
        """Reset instance id. Env should be a simulation environment."""
        pass

    @property
    @abstractmethod
    def instance_id_to_category_id(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def map_color_palette(self):
        pass

    @property
    @abstractmethod
    def frame_color_palette(self):
        pass

    @property
    @abstractmethod
    def categories_legend_path(self):
        pass

    @property
    @abstractmethod
    def num_sem_categories(self):
        pass

    @property
    def num_sem_obj_categories(self):
        return self.num_sem_categories()


class PaletteIndices:
    """
    Indices of different types of maps maintained in the agent's map state.
    """

    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    BEEN_CLOSE = 6
    SHORT_TERM_GOAL = 7
    BLACKLISTED_TARGETS_MAP = 8
    INSTANCE_BORDER = 9
    SEM_START = 10


# ----------------------------------------------------
# COCO Indoor Categories
# ----------------------------------------------------

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "no-category": 15,
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

coco_categories_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]

coco_categories_legend_path = str(
    Path(__file__).resolve().parent / "coco_categories_legend.png"
)

coco_frame_color_palette = [
    int(x * 255.0)
    for x in [
        *coco_categories_color_palette,
        1.0,
        1.0,
        1.0,  # no category
    ]
]

coco_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *coco_categories_color_palette,
    ]
]

mp3d_to_coco = {
    3: 0,  # chair
    10: 1,  # couch
    14: 2,  # plant
    11: 3,  # bed
    18: 4,  # toilet
    22: 5,  # tv
    5: 6,  # table
    15: 8,  # sink
}


class HM3DtoCOCOIndoor(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to COCO indoor
    category IDs.
    """

    def __init__(self):
        super().__init__()
        self.goal_id_to_goal_name = {idx: name for name, idx in coco_categories.items()}
        self.hm3d_goal_id_to_coco_goal_name = {
            0: "chair",
            1: "bed",
            2: "potted plant",
            3: "toilet",
            4: "tv",
            5: "couch",
        }
        self.hm3d_goal_id_to_coco_goal_id = {
            0: 0,  # chair
            1: 3,  # bed
            2: 2,  # potted plant
            3: 4,  # toilet
            4: 5,  # tv
            5: 1,  # couch
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_coco_goal_id[goal_id],
            self.hm3d_goal_id_to_coco_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, env):

        self._instance_id_to_category_id = np.array(
            [
                mp3d_to_coco.get(
                    hm3d_to_mp3d.get(obj.category.name().lower().strip()),
                    self.num_sem_categories - 1,
                )
                for obj in env.sim.semantic_annotations().objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return coco_map_color_palette

    @property
    def frame_color_palette(self):
        return coco_frame_color_palette

    @property
    def categories_legend_path(self):
        return coco_categories_legend_path

    @property
    def num_sem_categories(self):
        return 16


languagenav_2categories_indexes = {
    1: "target",
    2: "landmark",
}

languagenav_2categories_padded = (
    ["."] + [languagenav_2categories_indexes[i] for i in range(1, 3)] + ["other"]
)

languagenav_2categories_legend_path = str(
    Path(__file__).resolve().parent / "rearrange_3categories_legend.png"
)

# languagenav_2categories_color_palette = [255, 255, 255] + list(
#     d3_40_colors_rgb[1:3].flatten()
# )
languagenav_2categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:].flatten()
)
languagenav_2categories_frame_color_palette = languagenav_2categories_color_palette + [
    255,
    255,
    255,
]

languagenav_2categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *[x / 255.0 for x in languagenav_2categories_color_palette],
    ]
]


class LanguageNavCategories(SemanticCategoryMapping):
    """
    Mapping for LanguageNav episode visualizations and instance ID -> semantic category conversion.
    """

    def __init__(self):
        super().__init__()
        self.goal_id_to_goal_name = languagenav_2categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env):
        self._instance_id_to_category_id = []
        for obj in env.sim.semantic_annotations().objects:
            raw_category = obj.category.name().lower().strip()
            category = hm3d_raw_to_hm3d.get(raw_category)
            if category is None:
                self._instance_id_to_category_id.append(
                    self.all_hm3d_categories.index("unknown")
                )
            else:
                self._instance_id_to_category_id.append(
                    self.all_hm3d_categories.index(category)
                )

        self._instance_id_to_category_id = np.array(self._instance_id_to_category_id)

    @property
    def all_hm3d_categories(self):
        return list(set(all_hm3d_categories))

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return languagenav_2categories_map_color_palette

    @property
    def frame_color_palette(self):
        return languagenav_2categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return languagenav_2categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 is object category, 2 is start receptacle category, 3 is goal receptacle category, 4 is "other/misc"
        return 4


rearrange_3categories_indexes = {
    1: "object",
    2: "start_receptacle",
    3: "goal_receptacle",
}

rearrange_3categories_padded = (
    ["."] + [rearrange_3categories_indexes[i] for i in range(1, 4)] + ["other"]
)

rearrange_3categories_legend_path = str(
    Path(__file__).resolve().parent / "rearrange_3categories_legend.png"
)

rearrange_3categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:4].flatten()
)
rearrange_3categories_frame_color_palette = rearrange_3categories_color_palette + [
    255,
    255,
    255,
]


rearrange_3categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *[x / 255.0 for x in rearrange_3categories_color_palette],
    ]
]

# ----------------------------------------------------
# Mukul 33 Indoor Categories
# ----------------------------------------------------

mukul_33categories_indexes = {
    1: "alarm_clock",
    2: "bathtub",
    3: "bed",
    4: "book",
    5: "bottle",
    6: "bowl",
    7: "cabinet",
    8: "carpet",
    9: "chair",
    10: "chest_of_drawers",
    11: "couch",
    12: "cushion",
    13: "drinkware",
    14: "fireplace",
    15: "fridge",
    16: "laptop",
    17: "oven",
    18: "picture",
    19: "plate",
    20: "potted_plant",
    21: "shelves",
    22: "shoes",
    23: "shower",
    24: "sink",
    25: "stool",
    26: "table",
    27: "table_lamp",
    28: "toaster",
    29: "toilet",
    30: "tv",
    31: "vase",
    32: "wardrobe",
    33: "washer_dryer",
}
mukul_33categories_padded = (
    ["."] + [mukul_33categories_indexes[i] for i in range(1, 34)] + ["other"]
)

mukul_33categories_legend_path = str(
    Path(__file__).resolve().parent / "mukul_33categories_legend.png"
)

mukul_33categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:34].flatten()
)
mukul_33categories_frame_color_palette = mukul_33categories_color_palette + [
    255,
    255,
    255,
]

mukul_33categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *[x / 255.0 for x in mukul_33categories_color_palette],
    ]
]


class FloorplannertoMukulIndoor(SemanticCategoryMapping):
    """
    Mapping from category IDs in Floorplanner ObjectNav scenes/episodes to Mukul's 33
    indoor category IDs (semantic categories currently supported by Floorplanner, likely
    to evolve).
    """

    def __init__(self):
        super().__init__()
        self.floorplanner_goal_id_to_goal_name = mukul_33categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.floorplanner_goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env):
        # Identity everywhere except index 0 mapped to 34
        self._instance_id_to_category_id = np.arange(self.num_sem_categories)
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return mukul_33categories_map_color_palette

    @property
    def frame_color_palette(self):
        return mukul_33categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return mukul_33categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 to 33 are semantic categories, 34 is "other/misc"
        return 35


# ----------------------------------------------------
# HSSD 28 Indoor Categories
# ----------------------------------------------------

hssd_28categories_indexes = {
    1: "alarm_clock",
    2: "bed",
    3: "book",
    4: "bottle",
    5: "bowl",
    6: "chair",
    7: "chest_of_drawers",
    8: "couch",
    9: "cushion",
    10: "drinkware",
    11: "fridge",
    12: "laptop",
    13: "microwave",
    14: "picture",
    15: "plate",
    16: "potted_plant",
    17: "shelves",
    18: "shoes",
    19: "sink",
    20: "stool",
    21: "table",
    22: "table_lamp",
    23: "toaster",
    24: "toilet",
    25: "trashcan",
    26: "tv",
    27: "vase",
    28: "washer_dryer",
}

hssd_28categories_padded = (
    ["."] + [hssd_28categories_indexes[i] for i in range(1, 29)] + ["other"]
)

hssd_28categories_legend_path = str(
    Path(__file__).resolve().parent / "hssd_28_cat_legend.png"
)

hssd_28categories_color_palette = [255, 255, 255] + list(
    d3_40_colors_rgb[1:34].flatten()
)
hssd_28categories_frame_color_palette = hssd_28categories_color_palette + [
    255,
    255,
    255,
]

hssd_28categories_map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *[x / 255.0 for x in hssd_28categories_color_palette],
    ]
]


class HM3DtoHSSD28Indoor(SemanticCategoryMapping):
    """ """

    def __init__(self):
        super().__init__()
        self.floorplanner_goal_id_to_goal_name = hssd_28categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.floorplanner_goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env):
        """Reset habitat instance ids. Env should be a simulation environment."""
        pass

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return hssd_28categories_map_color_palette

    @property
    def frame_color_palette(self):
        return hssd_28categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return hssd_28categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 to 28 are semantic categories, 29 is "other/misc"
        return 30


class RearrangeBasicCategories(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self.goal_id_to_goal_name = rearrange_3categories_indexes
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env=None):
        """Reset instance IDs in habitat environments. Env should be a habitat env."""
        # Identity everywhere except index 0 mapped to 4
        self._instance_id_to_category_id = np.arange(self.num_sem_categories)
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return rearrange_3categories_map_color_palette

    @property
    def frame_color_palette(self):
        return rearrange_3categories_frame_color_palette

    @property
    def categories_legend_path(self):
        return rearrange_3categories_legend_path

    @property
    def num_sem_categories(self):
        # 0 is unused, 1 is object category, 2 is start receptacle category, 3 is goal receptacle category, 4 is "other/misc"
        return 5


rearrange_detic_categories_legend_path = str(
    Path(__file__).resolve().parent / "rearrange_detic_categories_legend.png"
)


class RearrangeDETICCategories(SemanticCategoryMapping):
    """Maintain category to id and category to color mappings for use in OVMM task.
    Uses a default list of categories if no category list is passed."""

    def __init__(self, categories_indexes, num_sem_objects=None):
        super().__init__()
        self.goal_id_to_goal_name = categories_indexes
        self._num_sem_obj_categories = num_sem_objects
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (goal_id, self.goal_id_to_goal_name[goal_id])

    def reset_instance_id_to_category_id(self, env=None):
        """Reset instance Ids."""
        self._instance_id_to_category_id = np.arange(self.num_sem_categories)
        self._instance_id_to_category_id[0] = self.num_sem_categories - 1

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def color_palette(self):
        color_palette = [255, 255, 255] + d3_40_colors_rgb[
            1 : self.num_sem_categories
        ].flatten().tolist()
        return color_palette

    @property
    def map_color_palette(self):
        map_color_palette = [
            int(x * 255.0)
            for x in [
                1.0,
                1.0,
                1.0,  # empty space
                0.6,
                0.6,
                0.6,  # obstacles
                0.95,
                0.95,
                0.95,  # explored area
                0.96,
                0.36,
                0.26,  # visited area
                0.12,
                0.46,
                0.70,  # closest goal
                0.63,
                0.78,
                0.95,  # rest of goal
                0.6,
                0.87,
                0.54,  # been close map
                0.0,
                1.0,
                0.0,  # short term goal
                0.6,
                0.17,
                0.54,  # blacklisted targets map
                0.0,
                0.0,
                0.0,  # instance border
                *[x / 255.0 for x in self.color_palette],
            ]
        ]

        return map_color_palette

    @property
    def frame_color_palette(self):
        frame_color_palette = self.color_palette + [
            255,
            255,
            255,
        ]
        return frame_color_palette

    @property
    def categories_legend_path(self):
        return rearrange_detic_categories_legend_path

    @property
    def num_sem_categories(self):
        return len(self.goal_id_to_goal_name.keys()) + 2

    @property
    def num_sem_obj_categories(self):
        return self._num_sem_obj_categories


# ----------------------------------------------------
# Long-tail Indoor Categories
# ----------------------------------------------------

receptacle_categories = [
    "armchair",
    "bar_chair",
    "bathroom_cabinet",
    "bathtub",
    "beanbag_chair",
    "bed",
    "bench",
    "bidet",
    "bookshelf",
    "cabinet",
    "chair",
    "clothes_hanger",
    "coat_rack",
    "coffee_machine",
    "coffee_table",
    "copier",
    "couch",
    "countertop",
    "desk",
    "dining_chair",
    "dining_table",
    "dish_rack",
    "dishwasher",
    "drawer",
    "end_table",
    "fireplace",
    "garbage_can",
    "hand_towel_holder",
    "highchair",
    "ironing_board",
    "kitchen_cabinet",
    "laundry_hamper",
    "massage_table",
    "microwave",
    "mixer",
    "nightstand",
    "office_chair",
    "ottoman",
    "oven",
    "pantry",
    "pool_table",
    "printer",
    "rack",
    "recycling_bin",
    "refrigerator",
    "safe",
    "shelving",
    "shower",
    "sink",
    "stool",
    "storage_container",
    "stove",
    "table",
    "television_stand",
    "toilet_paper_holder",
    "towel_holder",
    "washing_machine",
]
carryable_categories = [
    "alarm_clock",
    "aluminum_foil",
    "apple",
    "apron",
    "ashtray",
    "backpack",
    "bag",
    "ball",
    "baseball_bat",
    "basket",
    "basket_ball",
    "bath_mat",
    "bathrobe",
    "bedside_lamp",
    "beer",
    "belt",
    "blanket",
    "book",
    "bottle",
    "bowl",
    "box",
    "bread",
    "briefcase",
    "broom",
    "brush",
    "bucket",
    "butter_knife",
    "camera",
    "candle",
    "cd",
    "cell_phone",
    "chandelier",
    "chest",
    "cloth",
    "clothes",
    "coat",
    "coffee_kettle",
    "cosmetics",
    "credit_card",
    "cup",
    "cushion",
    "cutting_board",
    "detergent_bottle",
    "dining_table_mat",
    "dish_sponge",
    "doll",
    "dress",
    "drum",
    "dustpan",
    "egg",
    "exercise_mat",
    "firewood",
    "footrest",
    "fork",
    "fruit_bowl",
    "garbage_bag",
    "glass",
    "globe",
    "grocery_bag",
    "guitar",
    "guitar_case",
    "hair_dryer",
    "hand_towel",
    "hanger",
    "hat",
    "headphones",
    "jacket",
    "jar",
    "kettle",
    "key_chain",
    "keyboard",
    "knife",
    "ladle",
    "laptop",
    "laundry_detergent",
    "lettuce",
    "magazine",
    "mouse",
    "mug",
    "newspaper",
    "pan",
    "paper",
    "pencil",
    "pepper_shaker",
    "pillow",
    "pitcher",
    "plate",
    "platter",
    "plunger",
    "pot",
    "potato",
    "projector",
    "purse",
    "radio",
    "remote_control",
    "rope",
    "salt_shaker",
    "scale",
    "scarf",
    "scrub_brush",
    "shampoo",
    "sheet",
    "shirt",
    "shoe",
    "skateboard",
    "soap_bar",
    "soap_bottle",
    "soap_dish",
    "soda_can",
    "spatula",
    "speaker",
    "spoon",
    "spray_bottle",
    "stuffed_animal",
    "suitcase",
    "table_lamp",
    "tablecloth",
    "tablet",
    "teapot",
    "teddy_bear",
    "telephone",
    "tennis_racket",
    "tissue_box",
    "toilet_brush_holder",
    "toilet_paper",
    "tomato",
    "toolbox",
    "towel",
    "toy",
    "tray",
    "umbrella",
    "vase",
    "watch",
    "water_dispenser",
    "wine_bottle",
]
located_topography_categories = [
    "air_conditioner",
    "bathroom_stall",
    "bicycle",
    "blackboard",
    "clock",
    "desktop",
    "display_case",
    "door_mat",
    "fire_alarm",
    "fire_extinguisher",
    "floor_lamp",
    "fruit_bowl",
    "grill",
    "gym_equipment",
    "jacuzzi",
    "ladder",
    "monitor",
    "piano",
    "ping_pong_table",
    "plant",
    "potted_plant",
    "projector_screen",
    "radiator",
    "television",
    "toaster",
    "toilet",
    "toilet_paper_dispenser",
    "treadmill",
    "urinal",
    "vacuum_cleaner",
    "water_cooler",
    "whiteboard",
]
generic_topography_categories = [
    "arch",
    "blinds",
    "curtain",
    "door",
    "door_frame",
    "fan",
    "heater",
    "lamp",
    "light",
    "mirror",
    "outlet",
    "picture",
    "pillar",
    "poster",
    "rug",
    "stair_railing",
    "stairs",
    "statue",
    "tapestry",
    "thermostat",
    "window",
    "window_frame",
]
object_part_categories = [
    "door_handle",
    "door_knob",
    "faucet",
    "lightswitch",
    "shower_curtain",
    "shower_door",
    "shower_handle",
    "shower_head",
    "sink_basin",
    "stove_knob",
]
long_tail_indoor_categories = (
    receptacle_categories
    + carryable_categories
    + located_topography_categories
    + generic_topography_categories
    + object_part_categories
    + ["other"]
)
hm3d_to_longtail_indoor = {
    "wall": "other",
    "door": "rack",
    "ceiling": "other",
    "floor": "other",
    "picture": "door",
    "window": "other",
    "chair": "other",
    "frame": "other",
    "remove": "statue",
    "pillow": "ironing_board",
    "object": "other",
    "light": "other",
    "cabinet": "other",
    "curtain": "scale",
    "table": "cabinet",
    "plant": "other",
    "decoration": "wine_bottle",
    "lamp": "other",
    "mirror": "other",
    "towel": "other",
    "sink": "other",
    "shelf": "gym_equipment",
    "couch": "chest",
    "dining": "window_frame",
    "bed": "other",
    "nightstand": "picture",
    "toilet": "other",
    "sofa": "gym_equipment",
    "pillar": "cabinet",
    "handrail": "other",
    "stair": "other",
    "stool": "other",
    "armchair": "statue",
    "kitchen": "tray",
    "vase": "other",
    "cushion": "bathroom_cabinet",
    "tv": "drawer",
    "unknown": "end_table",
    "pot": "chair",
    "desk": "other",
    "roof": "picture",
    "box": "other",
    "shower": "window_frame",
    "coffee": "other",
    "countertop": "other",
    "bench": "blanket",
    "trashcan": "couch",
    "fireplace": "other",
    "clothes": "door",
    "bathtub": "sink_basin",
    "duct": "bench",
    "bath": "other",
    "book": "vase",
    "beam": "alarm_clock",
    "vent": "other",
    "faucet": "fireplace",
    "photo": "lamp",
    "paper": "other",
    "counter": "fireplace",
    "fan": "other",
    "step": "ashtray",
    "wash": "sink",
    "/otherroom": "other",
    "washbasin": "other",
    "railing": "door_knob",
    "shelving": "bucket",
    "statue": "stairs",
    "dresser": "other",
    "rug": "other",
    "ottoman": "monitor",
    "bottle": "picture",
    "office": "blinds",
    "refrigerator": "other",
    "bookshelf": "cosmetics",
    "end": "other",
    "wardrobe": "basket",
    "toiletry": "other",
    "pipe": "towel_holder",
    "monitor": "soap_dish",
    "stand": "other",
    "drawer": "other",
    "container": "exercise_mat",
    "switch": "other",
    "skylight": "rack",
    "purse": "picture",
    "doorway": "blinds",
    "paneling": "picture",
    "basket": "other",
    "closet": "other",
    "arch": "other",
    "chandelier": "rack",
    "oven": "stair_railing",
    "clock": "other",
    "footstool": "television_stand",
    "stove": "other",
    "washing": "other",
    "machine": "potted_plant",
    "rack": "shelving",
    "fire": "nightstand",
    "alarm": "other",
    "bin": "other",
    "chest": "tray",
    "microwave": "potted_plant",
    "blinds": "armchair",
    "bowl": "other",
    "tree": "gym_equipment",
    "vanity": "other",
    "tissue": "light",
    "plate": "other",
    "shoe": "other",
    "heater": "other",
    "bedframe": "coffee_table",
    "headboard": "other",
    "post": "shelving",
    "swivel": "other",
    "pedestal": "printer",
    "fence": "other",
    "pew": "other",
    "bucket": "other",
    "decorative": "sheet",
    "mask": "shower",
    "candle": "jar",
    "flowerpot": "door_mat",
    "speaker": "other",
    "seat": "toy",
    "sign": "door",
    "air": "other",
    "conditioner": "monitor",
    "rod": "other",
    "clutter": "other",
    "extinguisher": "mirror",
    "mat": "cosmetics",
    "sculpture": "other",
    "printer": "end_table",
    "telephone": "other",
    "molding": "other",
    "handbag": "arch",
    "blanket": "gym_equipment",
    "dispenser": "other",
    "handle": "potted_plant",
    "/outside": "mug",
    "screen": "other",
    "showerhead": "washing_machine",
    "barricade": "desk",
    "soap": "other",
    "banister": "other",
    "keyboard": "statue",
    "thermostat": "scale",
    "radiator": "garbage_can",
    "island": "copier",
    "dryer": "other",
    "panel": "end_table",
    "glass": "other",
    "dishwasher": "towel_holder",
    "cup": "other",
    "bathroom": "other",
    "ladder": "other",
    "garage": "other",
    "hat": "cabinet",
    "of": "other",
    "drawers": "bench",
    "exit": "other",
    "side": "other",
    "piano": "cabinet",
    "board": "other",
    "archway": "cabinet",
    "rope": "floor_lamp",
    "ball": "laundry_hamper",
    "gym": "fireplace",
    "equipment": "tissue_box",
    "hanger": "other",
    "easy": "other",
    "lounge": "bottle",
    "furniture": "box",
    "carpet": "other",
    "food": "other",
    "ridge": "other",
    "candlestick": "other",
    "computer": "other",
    "sconce": "other",
    "scale": "other",
    "baseboard": "toilet",
    "bag": "other",
    "laptop": "bathroom_cabinet",
    "treadmill": "dress",
    "staircase": "water_dispenser",
    "guitar": "door",
    "fixture": "other",
    "display": "table",
    "case": "gym_equipment",
    "exercise": "other",
    "holder": "cabinet",
    "basin": "shower",
    "bar": "other",
    "tray": "window",
    "urn": "other",
    "shade": "toilet",
    "grass": "other",
    "pool": "toaster",
    "coat": "other",
    "cloth": "other",
    "water": "other",
    "cooler": "other",
    "ledge": "arch",
    "utensil": "box",
    "shrubbery": "shelving",
    "teapot": "coffee_machine",
    "locker": "other",
    "ornament": "refrigerator",
    "bidet": "suitcase",
    "window/door": "laundry_hamper",
    "stuffed": "door_frame",
    "animal": "other",
    "fencing": "tablet",
    "lampshade": "exercise_mat",
    "bust": "other",
    "car": "other",
    "figure": "display_case",
    "set": "mouse",
    "brush": "other",
    "doll": "jar",
    "drum": "gym_equipment",
    "dress": "shelving",
    "whiteboard": "jacuzzi",
    "opener": "fireplace",
    "range": "window_frame",
    "hood": "toilet",
    "easel": "stool",
    "fruit": "other",
    "appliance": "other",
    "candelabra": "other",
    "toy": "stair_railing",
    "top": "other",
    "highchair": "other",
    "footrest": "clothes",
    "dish": "bed",
    "altar": "picture",
    "place": "table",
    "sheet": "chair",
    "wood": "grill",
    "robe": "cabinet",
    "stall": "dining_table",
    "plush": "door_knob",
    "bush": "other",
    "valence": "other",
    "control": "grill",
    "tap": "arch",
    "shampoo": "storage_container",
    "massage": "shelving",
    "knob": "fan",
    "stopper": "other",
    "bulletin": "other",
    "electric": "other",
    "wire": "other",
    "casing": "door",
    "storage": "other",
    "maker": "other",
    "projector": "statue",
    "cubby": "washing_machine",
    "balcony": "couch",
    "/w": "other",
    "pan": "other",
    "luggage": "other",
    "hamper": "other",
    "trinket": "other",
    "backsplash": "other",
    "chimney": "door_frame",
    "person": "other",
    "tablet": "other",
    "smoke": "other",
    "weight": "pillar",
    "bedpost": "other",
    "file": "gym_equipment",
    "umbrella": "bedside_lamp",
    "laundry": "other",
    "jar": "urinal",
    "bike": "other",
    "hose": "other",
    "dormer": "firewood",
    "power": "rug",
    "breaker": "picture",
    "detector": "pillow",
    "jacuzzi": "other",
    "backpack": "stove",
    "hook": "oven",
    "elevator": "mirror",
    "tool": "recycling_bin",
    "recliner": "countertop",
    "recessed": "other",
    "tank": "other",
    "toaster": "other",
    "landing": "door_frame",
    "hunting": "bicycle",
    "trophy": "copier",
    "motion": "kitchen_cabinet",
    "can": "other",
    "paint": "other",
    "medicine": "other",
    "sensor": "lightswitch",
    "cart": "door_frame",
    "slab": "other",
    "bean": "clock",
    "pole": "other",
    "canister": "other",
    "pitcher": "other",
    "podium": "mirror",
    "grill": "other",
    "tapestry": "other",
    "doorknob": "other",
    "vacuum": "detergent_bottle",
    "cleaner": "other",
    "comforter": "other",
    "shirt": "other",
    "dressing": "jacket",
    "beside": "nightstand",
    "curb": "other",
    "support": "other",
    "globe": "chair",
    "pantry": "other",
    "skateboard": "other",
    "cabin": "light",
    "chaise": "door",
    "flower": "curtain",
    "and": "other",
    "chairs": "blinds",
    "cross": "water_dispenser",
    "sliding": "other",
    "cosmetics": "bench",
    "kettle": "other",
    "junk": "other",
    "stationery": "office_chair",
    "gate": "other",
    "safe": "other",
    "ventilation": "other",
    "firewood": "statue",
    "row": "other",
    "theater": "other",
    "toolbox": "speaker",
    "security": "stair_railing",
    "camera": "nightstand",
    "mantle": "other",
    "skirting": "stairs",
    "tile": "other",
    "outlet": "other",
    "doorframe": "other",
    "hedge": "outlet",
    "hand": "other",
    "christmas": "window",
    "column": "other",
    "casket": "blackboard",
    "centerpiece": "lamp",
    "bedside": "other",
    "item": "other",
    "fountain": "other",
    "soffit": "other",
    "urinal": "other",
    "barrel": "shelving",
    "roll": "other",
    "portrait": "chair",
    "pouffe": "other",
    "concrete": "other",
    "block": "bar_chair",
    "liner": "other",
    "patio": "other",
    "folding": "sink",
    "recycle": "shelving",
    "rafter": "chair",
    "stage": "other",
    "sprinkler": "monitor",
    "soil": "other",
    "bicycle": "other",
    "partition": "table",
    "led": "desktop",
    "under": "other",
    "books": "picture",
    "giraffe": "door_frame",
    "grandfather": "other",
    "jewelry": "other",
    "bottles": "other",
    "wine": "stair_railing",
    "dog": "briefcase",
    "valance": "door",
    "radio": "other",
    "seats": "other",
    "towels": "other",
    "sauna": "desktop",
    "fume": "cloth",
    "cupboard": "vacuum_cleaner",
    "mouse": "other",
    "boiler": "other",
    "hearth": "other",
    "round": "shower",
    "doorstep": "other",
    "binder": "chandelier",
    "runner": "other",
    "cubicle": "other",
    "overhang": "door",
    "bathrobe": "other",
    "doormat": "other",
    "jacket": "shelving",
    "trim": "other",
    "reflection": "stairs",
    "pulpit": "other",
    "armchairs": "door",
    "fish": "other",
    "objects": "tray",
    "lintel": "other",
    "lighting": "cloth",
    "freezer": "table",
    "extractor": "footrest",
    "platform": "cabinet",
    "hot": "projector",
    "tub": "other",
    "grab": "bathtub",
    "detail": "other",
    "whine": "cabinet",
    "painting": "bench",
    "buffet": "other",
    "billow": "other",
    "stairs": "other",
    "calendar": "other",
    "dome": "other",
    "poll": "shower_handle",
    "wet": "statue",
    "stovetop": "desktop",
    "vending": "door",
    "liquid": "stairs",
    "small": "other",
    "table/stand": "paper",
    "shutters": "other",
    "stone": "other",
    "tripod": "door_frame",
    "wreath": "other",
    "hinge": "cabinet",
    "french": "other",
    "night": "other",
    "picure": "chair",
    "stick": "stool",
    "fluorescent": "other",
    "trellis": "beanbag_chair",
    "dartboard": "other",
    "dirt": "other",
    "base": "bathtub",
    "chemical": "apron",
    "misc": "ironing_board",
    "cover": "other",
    "reading": "other",
    "steps": "end_table",
    "sideboard": "other",
    "separator": "chair",
    "vessel": "statue",
    "skirt": "couch",
    "rocking": "other",
    "blackboard": "chair",
    "closest": "other",
    "area": "other",
    "scroll": "other",
    "foot": "desk",
    "button": "bathtub",
    "art/clutter": "other",
    "shovel": "toilet_paper",
    "yard": "other",
    "semi": "other",
    "bouquet": "other",
    "corner": "other",
    "plunger": "potted_plant",
    "belt": "other",
    "sewing": "other",
    "water/cold": "other",
    "barbecue": "other",
    "cutting": "curtain",
    "soapbox": "mixer",
    "stuff": "statue",
    "copier": "bench",
    "picture/window": "shelving",
    "throne": "platter",
    "socket": "other",
    "art": "shelving",
    "tabletop": "bathrobe",
    "trash": "cushion",
    "l-shaped": "clothes",
    "cardboard": "towel",
    "hanging": "fire_extinguisher",
    "stand/small": "other",
    "indent": "stairs",
    "towel/curtain": "other",
    "iron": "other",
    "shelf/cabinet": "nightstand",
    "accessory": "other",
    "circular": "other",
    "dustpan": "other",
    "oil": "other",
    "scaffolding": "other",
    "baluster": "refrigerator",
    "leg": "stairs",
    "rest": "garbage_can",
    "/otheroom": "other",
    "hole": "pillow",
    "ping": "other",
    "pong": "towel",
    "hutch": "refrigerator",
    "foliage": "picture",
    "circle": "plant",
    "record": "window",
    "player": "window",
    "doorpost": "other",
    "briefcase": "towel_holder",
    "energy": "pillow",
    "beanbag": "door",
    "plumbing": "other",
    "moose": "toy",
    "head/sculpture/hunting": "toilet",
    "flowerbed": "other",
    "antique": "other",
    "rock": "light",
    "caddy": "faucet",
    "media": "other",
    "console": "other",
    "risers": "other",
    "for": "other",
    "seating": "other",
    "branch": "other",
    "tiled": "vase",
    "bedroom": "other",
    "hearst": "basket",
    "condiment": "other",
    "piping": "window",
    "shelves": "other",
    "watch": "other",
    "rail": "other",
    "fuse": "other",
    "knife": "other",
    "aquarium": "fire_alarm",
    "wheelbarrow": "armchair",
    "rods/table": "other",
    "gable": "other",
    "balustrade": "other",
    "three": "other",
    "rocky": "other",
    "ground": "other",
    "backrest": "other",
    "basketball": "toy",
    "hoop": "other",
    "spice": "other",
    "cluttered": "other",
    "transformer": "door",
    "gift": "table",
    "stack": "other",
    "papers": "shower",
    "holy": "other",
    "arcade": "other",
    "game": "fireplace",
    "-": "shower_head",
    "probably": "couch",
    "part": "other",
    "--": "other",
    "maybe": "other",
    "compound": "other",
    "plug": "other",
    "magazine": "floor_lamp",
    "rolling": "statue",
    "pin": "other",
    "sink/basin": "nightstand",
    "boarder": "other",
    "perfume": "other",
    "heat": "cabinet",
    "pump": "exercise_mat",
    "columned": "shower_curtain",
    "perimeter": "other",
    "shrine": "other",
    "canvas": "other",
    "art/man": "chair",
    "credenza": "other",
    "artwork": "other",
    "playpen": "other",
    "makeup": "other",
    "plant/art": "lamp",
    "bot": "door_frame",
    "horse": "other",
    "lower": "television_stand",
    "pack": "other",
    "pathway": "other",
    "tablecloth": "other",
    "tarp": "other",
    "clothing": "other",
}


class HM3DtoLongTailIndoor(SemanticCategoryMapping):
    def __init__(self):
        super().__init__()
        self.hm3d_goal_id_to_longtail_goal_name = {
            0: "chair",
            1: "bed",
            2: "potted plant",
            3: "toilet",
            4: "tv",
            5: "couch",
        }
        self.hm3d_goal_id_to_longtail_goal_id = {
            0: long_tail_indoor_categories.index("chair"),
            1: long_tail_indoor_categories.index("bed"),
            2: long_tail_indoor_categories.index("potted_plant"),
            3: long_tail_indoor_categories.index("toilet"),
            4: long_tail_indoor_categories.index("television"),
            5: long_tail_indoor_categories.index("couch"),
        }
        self._instance_id_to_category_id = None

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        return (
            self.hm3d_goal_id_to_longtail_goal_id[goal_id],
            self.hm3d_goal_id_to_longtail_goal_name[goal_id],
        )

    def reset_instance_id_to_category_id(self, env):
        """Reset instance IDs from semantic annotations. Env here should be a simulation env from habitat."""
        self._instance_id_to_category_id = np.ndarray(
            [
                long_tail_indoor_categories.index(
                    hm3d_to_longtail_indoor.get(
                        obj.category.name().lower().strip(), "other"
                    )
                )
                for obj in env.sim.semantic_annotations().objects
            ]
        )

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        # TODO Replace with appropriate color palette
        return coco_map_color_palette

    @property
    def frame_color_palette(self):
        # TODO Replace with appropriate color palette
        return coco_frame_color_palette

    @property
    def categories_legend_path(self):
        # TODO Replace with appropriate legend
        return coco_categories_legend_path

    @property
    def num_sem_categories(self):
        return len(long_tail_indoor_categories)
