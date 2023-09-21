# If True, use dummy offline environment that loads observations from disk
OFFLINE = False

import json
import math
import pickle
import pprint
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

from collections import defaultdict

import cv2
import numpy as np
import skimage.morphology
import torch
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image, ImageDraw, ImageFont

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

if not OFFLINE:
    from spot_wrapper.spot import Spot

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories_color_palette,
    coco_category_id_to_coco_category,
)
from home_robot.utils.config import get_config
from home_robot_hw.env.spot_goat_env import SpotGoatEnv
from home_robot_hw.env.spot_goat_offline_env import SpotGoatOfflineEnv


class PI:
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    SHORT_TERM_GOAL = 6
    SEM_START = 7


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


def resize_image_to_fit(img, target_width, target_height):
    # Calculate the aspect ratio of the original image.
    original_width, original_height = img.shape[1], img.shape[0]
    aspect_ratio = original_width / original_height

    # Determine the dimensions to which the image should be resized.
    if original_width > original_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
        if new_height > target_height:
            new_height = target_height
            new_width = int(aspect_ratio * new_height)
    else:
        new_height = target_height
        new_width = int(aspect_ratio * new_height)
        if new_width > target_width:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

    # Resize the image.
    resized_img = cv2.resize(img, (new_width, new_height))

    # If you want to ensure the resulting image is exactly the target size,
    # create a blank canvas and paste the resized image onto it.
    canvas = (
        np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    )  # Assuming white canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[
        y_offset : y_offset + new_height, x_offset : x_offset + new_width
    ] = resized_img

    return canvas


def generate_legend(
    vis_image: np.ndarray,
    colors: np.ndarray,
    texts: List[str],
    start_x: int,
    start_y: int,
    total_w: int,
    total_h: int,
):
    font = 0
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    # grid size - number of labels in each column/row
    grid_w, grid_h = 7, 6
    int_w = total_w / grid_w
    int_h = total_h / grid_h
    ctr = 0
    for y in range(grid_h):
        for x in range(grid_w):
            if ctr > len(colors) - 1:
                break
            rect_start_x = int(total_w * x / grid_w) + start_x
            rect_start_y = int(total_h * y / grid_h) + start_y
            rect_start = [rect_start_x, rect_start_y]
            rect_end_x = rect_start_x + int(int_h * 0.2) + 20
            rect_end_y = rect_start_y + int(int_h * 0.2) + 10
            rect_end = [rect_end_x, rect_end_y]
            vis_image = cv2.rectangle(
                vis_image, rect_start, rect_end, colors[ctr].tolist(), thickness=-1
            )
            vis_image = cv2.putText(
                vis_image,
                texts[ctr],
                (rect_end_x + 5, rect_end_y - 5),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA,
            )
            ctr += 1
    return vis_image


def get_semantic_map_vis(
    semantic_map: Categorical2DSemanticMapState,
    semantic_frame: np.array,
    closest_goal_map: np.array,
    # depth_frame: np.array,
    goal_image: np.array,
    instance_image: Optional[np.array] = None,
    instance_memory: Optional[InstanceMemory] = None,
    visualize_instances: bool = False,
    legend=None,
    visualize_goal=True,
    subgoal=None,
):
    vis_image = np.ones((655, 1820, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Segmentation"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(
        vis_image,
        text,
        (textX, textY),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # text = "Depth"
    text = "Goal"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (640 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(
        vis_image,
        text,
        (textX, textY),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    text = "Predicted Instance Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 1280 + (480 - textsize[0]) // 2 + 45
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(
        vis_image,
        text,
        (textX, textY),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    if legend is None:
        map_color_palette = [
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
            0.00,
            0.00,
            0.00,  # short term goal
        ]
        map_color_palette = [int(x * 255.0) for x in map_color_palette]
        map_color_palette += d3_40_colors_rgb.flatten().tolist()

        new_colors = d3_40_colors_rgb.copy()
        new_colors[:, 0] = np.minimum(new_colors[:, 0] + 15, 255)
        new_colors[:, 2] = np.maximum(new_colors[:, 2] - 15, 0)
        map_color_palette += new_colors.flatten().tolist()
    else:
        map_color_palette = [
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
            0.00,
            0.00,
            0.00,  # short term goal
            *coco_categories_color_palette,
        ]
        map_color_palette = [int(x * 255.0) for x in map_color_palette]

    semantic_categories_map = semantic_map.get_semantic_map(0)
    obstacle_map = semantic_map.get_obstacle_map(0)
    explored_map = semantic_map.get_explored_map(0)
    visited_map = semantic_map.get_visited_map(0)
    goal_map = semantic_map.get_goal_map(0)
    instance_map = semantic_map.get_instance_map(0)

    no_category_mask = semantic_categories_map == semantic_map.num_sem_categories - 1
    if not visualize_instances:
        semantic_categories_map += PI.SEM_START
    else:
        unique_instances, remapped_instances = np.unique(
            instance_map, return_inverse=True
        )

        # project instance map
        projected_instance_map = instance_map.max(0)

        semantic_categories_map = projected_instance_map
        semantic_categories_map += PI.SEM_START - 1
        semantic_categories_map[
            semantic_categories_map == PI.SEM_START - 1
        ] = PI.EMPTY_SPACE

        num_instances = int(np.max(unique_instances))

        if num_instances > 2 * len(d3_40_colors_rgb):
            raise NotImplementedError

    obstacle_mask = np.rint(obstacle_map) == 1
    explored_mask = np.rint(explored_map) == 1
    visited_mask = visited_map == 1
    semantic_categories_map[no_category_mask] = PI.EMPTY_SPACE
    semantic_categories_map[
        np.logical_and(no_category_mask, explored_mask)
    ] = PI.EXPLORED
    semantic_categories_map[
        np.logical_and(no_category_mask, obstacle_mask)
    ] = PI.OBSTACLES
    semantic_categories_map[visited_mask] = PI.VISITED

    # Goal
    selem = skimage.morphology.disk(4)
    goal_mat = (1 - skimage.morphology.binary_dilation(goal_map, selem)) != 1
    goal_mask = goal_mat == 1
    semantic_categories_map[goal_mask] = PI.REST_OF_GOAL
    if closest_goal_map is not None:
        closest_goal_mat = (
            1 - skimage.morphology.binary_dilation(closest_goal_map, selem)
        ) != 1
        closest_goal_mask = closest_goal_mat == 1
        semantic_categories_map[closest_goal_mask] = PI.CLOSEST_GOAL
    if subgoal is not None:
        subgoal_map = np.zeros_like(goal_map)
        # might need to flip row value
        subgoal_map[subgoal[0], subgoal[1]] = 1
        subgoal_mat = (1 - skimage.morphology.binary_dilation(subgoal_map, selem)) != 1
        subgoal_mask = subgoal_mat == 1
        semantic_categories_map[subgoal_mask] = PI.SHORT_TERM_GOAL

    # Draw semantic map
    semantic_map_vis = Image.new("P", semantic_categories_map.shape)
    semantic_map_vis.putpalette(map_color_palette)
    semantic_map_vis.putdata(semantic_categories_map.flatten().astype(np.uint8))
    semantic_map_vis = semantic_map_vis.convert("RGB")
    semantic_map_vis = np.flipud(semantic_map_vis)
    semantic_map_vis = cv2.resize(
        semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
    )

    vis_image[50:530, 1325:1805] = semantic_map_vis

    # Draw semantic frame
    vis_image[50:530, 15:655] = cv2.resize(semantic_frame[:, :, ::-1], (640, 480))
    # vis_image[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

    # Draw depth frame
    # vis_image[50:530, 670:1310] = cv2.resize(depth_frame, (640, 480))

    # Draw goal image
    vis_image[50:530, 670:1310] = resize_image_to_fit(goal_image, 640, 480)

    # Draw legend
    if legend is not None:
        lx, ly, _ = legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend[:, :, ::-1]
    elif visualize_instances:
        # Name instances as chair-1, chair-2 and so on
        category_counts = defaultdict(int)
        instance_to_name = {}
        for instance in range(1, int(np.max(unique_instances)) + 1):
            if instance == 0:
                continue
            if instance_memory is not None:
                # retrieve name
                category = instance_memory.instances[0][int(instance)].category_id
                category_counts[category] += 1
                instance_to_name[instance] = (
                    coco_category_id_to_coco_category[category]
                    + f" - {category_counts[category]}"
                )
            else:
                instance_to_name[instance] = f"Instance - {instance}"

        vis_image = generate_legend(
            vis_image,
            np.array(
                map_color_palette[3 * PI.SEM_START : (PI.SEM_START + num_instances) * 3]
            ).reshape(-1, 3),
            [instance_to_name[i] for i in range(1, num_instances + 1)],
            155,
            537,
            1250,
            115,
        )

    # Draw agent arrow
    curr_x, curr_y, curr_o, gy1, _, gx1, _ = semantic_map.get_planner_pose_inputs(0)
    pos = (
        (curr_x * 100.0 / semantic_map.resolution - gx1)
        * 480
        / semantic_map.local_map_size,
        (semantic_map.local_map_size - curr_y * 100.0 / semantic_map.resolution + gy1)
        * 480
        / semantic_map.local_map_size,
        np.deg2rad(-curr_o),
    )
    agent_arrow = vu.get_contour_points(pos, origin=(1325, 50), size=10)
    color = map_color_palette[9:12]
    cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

    return vis_image


def text_to_image(
    text,
    width,
    height,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
):
    # Create a blank image with the specified dimensions
    image = Image.new(
        "RGB", (width, height), color=(73, 109, 137)
    )  # RGB color can be any combination you like
    # Set up the drawing context
    d = ImageDraw.Draw(image)
    # Set the font and size. Font path might be different in your system. Install a font if necessary.
    font = ImageFont.truetype(font_path, 15)
    # Calculate width and height of the text to center it
    text_width, text_height = d.textsize(text, font=font)
    position = ((width - text_width) / 2, (height - text_height) / 2)
    # Add the text to the image
    d.text(position, text, fill=(255, 255, 255), font=font)
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    return image_array


# Fremont goals
GOALS = {
    # Object goals
    "object_chair": {"type": "objectnav", "target": "chair"},
    "object_couch": {"type": "objectnav", "target": "couch"},
    "object_plant": {"type": "objectnav", "target": "potted plant"},
    "object_bed": {"type": "objectnav", "target": "bed"},
    "object_toilet": {"type": "objectnav", "target": "toilet"},
    "object_tv": {"type": "objectnav", "target": "tv"},
    "object_table": {"type": "objectnav", "target": "dining table"},
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


# Example command:
# python projects/spot/goat.py --trajectory=trajectory1 --goals=object_toilet,image_bed1,language_chair4,image_couch1,language_plant1,language_plant2,image_refrigerator1
def main(spot=None, args=None):
    config_path = "projects/spot/configs/config.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.DUMP_LOCATION = (
        f"{str(Path(__file__).resolve().parent)}/fremont_trajectories/{args.trajectory}"
    )
    config.freeze()

    output_visualization_dir = f"{config.DUMP_LOCATION}/main_visualization"
    Path(output_visualization_dir).mkdir(parents=True, exist_ok=True)

    # create instance_memory and semantic_map dirs
    Path(f"{config.DUMP_LOCATION}/instance_memory").mkdir(parents=True, exist_ok=True)
    Path(f"{config.DUMP_LOCATION}/semantic_map").mkdir(parents=True, exist_ok=True)

    obs_dir = f"{config.DUMP_LOCATION}/obs"
    Path(obs_dir).mkdir(parents=True, exist_ok=True)

    legend_path = f"{str(Path(__file__).resolve().parent)}/coco_categories_legend.png"
    legend = cv2.imread(legend_path)
    vis_images = []

    if OFFLINE:
        # Load observations from disk instead of generating them
        env = SpotGoatOfflineEnv()
        env.reset(obs_dir)

    else:
        env = SpotGoatEnv(
            spot,
            position_control=True,
            estimated_depth_threshold=config.ENVIRONMENT.estimated_depth_threshold,
        )
        env.reset()

    goal_strings = args.goals.split(",")
    print("Goals:", goal_strings)
    goals = [GOALS.get(g) for g in goal_strings]
    goals = [g for g in goals if g is not None]
    env.set_goals(goals)
    with open(f"{config.DUMP_LOCATION}/goals.json", "w") as f:
        json.dump(goal_strings, f, indent=4)

    agent = GoatAgent(config=config)
    agent.reset()

    pan_warmup = False
    keyboard_takeover = args.keyboard
    if pan_warmup:
        assert not OFFLINE
        positions = spot.get_arm_joint_positions()
        new_pos = positions.copy()
        new_pos[0] = np.pi
        spot.set_arm_joint_positions(new_pos, travel_time=3)
        time.sleep(3)

    global_start_time = time.time()
    t = 0
    while not env.episode_over:
        step_start_time = time.time()
        t += 1
        print()
        print("STEP =", t)
        print("Subgoal step =", agent.sub_task_timesteps[0][agent.current_task_idx])
        print(f"Time: {step_start_time - global_start_time:.2f}")
        print(f"Goal {agent.current_task_idx}: {goal_strings[agent.current_task_idx]}")

        if not OFFLINE:
            obs = env.get_observation()
            obs.task_observations["current_task_idx"] = agent.current_task_idx
            obs.task_observations["timestamp"] = step_start_time - global_start_time
            print(
                f"Environment (including segmentation) {time.time() - step_start_time:2f}"
            )
            with open(f"{obs_dir}/{t}.pkl", "wb") as f:
                pickle.dump(obs, f)
        else:
            try:
                obs = env.get_observation(t)
            except FileNotFoundError:
                print(f"Could not load obs {t}")
                break

        action, info = agent.act(obs)
        print(f"Step time {time.time() - step_start_time:2f}")
        # print("SHORT_TERM:", info["short_term_goal"])
        x, y = info["short_term_goal"]
        x, y = agent.semantic_map.local_to_global(x, y)
        action = ContinuousNavigationAction(np.array([x, y, 0.0]))

        # Visualize map
        # depth_frame = obs.depth
        # if depth_frame.max() > 0:
        #     depth_frame = depth_frame / depth_frame.max()
        # depth_frame = (depth_frame * 255).astype(np.uint8)
        # depth_frame = np.repeat(depth_frame[:, :, np.newaxis], 3, axis=2)

        current_goal = goals[agent.current_task_idx]
        if current_goal["type"] == "imagenav":
            goal_image = current_goal["image"][:, :, ::-1]
        elif current_goal["type"] == "languagenav":
            goal_image = text_to_image(current_goal["instruction"], 640, 480)
        elif current_goal["type"] == "objectnav":
            goal_image = text_to_image(current_goal["target"], 640, 480)

        vis_image = get_semantic_map_vis(
            agent.semantic_map,
            semantic_frame=obs.task_observations["semantic_frame"],
            closest_goal_map=info["closest_goal_map"],
            subgoal=info["short_term_goal"],
            # depth_frame,
            goal_image=goal_image,
            legend=legend
            if (not config.AGENT.SEMANTIC_MAP.record_instance_ids)
            else None,
            instance_memory=agent.instance_memory,
            visualize_instances=config.AGENT.SEMANTIC_MAP.record_instance_ids,
        )
        instance_mem = agent.instance_memory
        with open(
            f"{config.DUMP_LOCATION}/instance_memory/instance_memory_{t}.pkl", "wb"
        ) as f:
            pickle.dump(instance_mem, f)

        # Save semantic map
        semantic_map = agent.semantic_map
        with open(
            f"{config.DUMP_LOCATION}/semantic_map/semantic_map_{t}.pth", "wb"
        ) as f:
            torch.save(semantic_map, f)

        vis_images.append(vis_image)
        cv2.imwrite(f"{output_visualization_dir}/{t}.png", vis_image[:, :, ::-1])

        if not OFFLINE:
            cv2.imshow("vis", vis_image[:, :, ::-1])
            cv2.imshow("depth", obs.depth / obs.depth.max())
            key = cv2.waitKey(1)

            if key == ord("z"):
                break

            if key == ord("q"):
                keyboard_takeover = True
                print("KEYBOARD TAKEOVER")

            if keyboard_takeover:
                if key == ord("w"):
                    spot.set_base_velocity(0.5, 0, 0, 0.5)
                elif key == ord("s"):
                    # back
                    spot.set_base_velocity(-0.5, 0, 0, 0.5)
                elif key == ord("a"):
                    # left
                    spot.set_base_velocity(0, 0.5, 0, 0.5)
                elif key == ord("d"):
                    # right
                    spot.set_base_velocity(0, -0.5, 0, 0.5)

            else:
                if pan_warmup:
                    positions = spot.get_arm_joint_positions()
                    new_pos = positions.copy()
                    new_pos[0] = -np.pi
                    spot.set_arm_joint_positions(new_pos, travel_time=20)
                    if positions[0] < -2.5:
                        pan_warmup = False
                        env.env.initialize_arm()
                else:
                    if action is not None:
                        env.apply_action(action)

    out_dest = f"{output_visualization_dir}/video.mp4"
    print("Writing", out_dest)
    create_video(
        [v[:, :, ::-1] for v in vis_images],
        out_dest,
        fps=5,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--trajectory", default="trajectory1")
    parser.add_argument("--goals", default="object_chair,object_sink")
    parser.add_argument("--keyboard", action="store_true")

    args = parser.parse_args()

    if not OFFLINE:
        spot = Spot("RealNavEnv")
        with spot.get_lease(hijack=True):
            main(spot, args)
    else:
        main()
