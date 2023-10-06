import math
import pickle
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import skimage.morphology
from matplotlib import pyplot as plt
from PIL import Image
from spot_wrapper.depth_utils import point_from_depth_image

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from spot_wrapper.spot import BODY_FRAME_NAME, VISION_FRAME_NAME, Spot

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.utils.config import get_config
from home_robot_hw.env.spot_objectnav_env import SpotObjectNavEnv


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


def get_semantic_map_vis(
    semantic_map: Categorical2DSemanticMapState,
    semantic_frame: np.array,
    closest_goal_map: np.array,
    depth_frame: np.array,
    color_palette: List[float],
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

    text = "Depth"
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

    text = "Predicted Semantic Map"
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

    map_color_palette = [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        1.00,
        0.90,
        0.90,  # explored area
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
        *color_palette,
    ]
    map_color_palette = [int(x * 255.0) for x in map_color_palette]

    semantic_categories_map = semantic_map.get_semantic_map(0)
    obstacle_map = semantic_map.get_obstacle_map(0)
    explored_map = semantic_map.get_explored_map(0)
    visited_map = semantic_map.get_visited_map(0)
    goal_map = semantic_map.get_goal_map(0)

    semantic_categories_map += PI.SEM_START
    no_category_mask = (
        semantic_categories_map == PI.SEM_START + semantic_map.num_sem_categories - 1
    )
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
    if visualize_goal:
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
            subgoal_mat = (
                1 - skimage.morphology.binary_dilation(subgoal_map, selem)
            ) != 1
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
    vis_image[50:530, 670:1310] = cv2.resize(depth_frame, (640, 480))

    # Draw legend
    if legend is not None:
        lx, ly, _ = legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend[:, :, ::-1]

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


def main(spot, args):
    config_path = "projects/spot/configs/config.yaml"
    config, config_str = get_config(config_path)

    output_visualization_dir = (
        f"{str(Path(__file__).resolve().parent)}/map_visualization/"
    )
    Path(output_visualization_dir).mkdir(parents=True, exist_ok=True)

    obs_dir = f"{str(Path(__file__).resolve().parent)}/obs/"
    Path(obs_dir).mkdir(parents=True, exist_ok=True)

    legend_path = f"{str(Path(__file__).resolve().parent)}/coco_categories_legend.png"
    legend = cv2.imread(legend_path)
    vis_images = []

    env = SpotObjectNavEnv(spot, position_control=True)
    env.reset()
    if args.category:
        env.set_goal(args.category)
    else:
        user_input = input("Enter the goal category: ")
        print("You entered:", user_input)
        env.set_goal(user_input)

    agent = ObjectNavAgent(config=config)
    agent.reset()

    assert agent.num_sem_categories == env.num_sem_categories
    pan_warmup = args.rotate

    # control with keyboard instead of planner
    keyboard_takeover = args.keyboard
    if pan_warmup:
        env.env.set_arm_yaw(np.pi + np.pi / 4, time=4, blocking=True)
        positions = spot.get_arm_joint_positions()
        print(positions)

    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)

        obs = env.get_observation()
        with open(f"{obs_dir}/{t}.pkl", "wb") as f:
            pickle.dump(obs, f)

        action, info = agent.act(obs)
        # env.env.set_arm_yaw(np.pi/4)
        # import transforms3d as t3d
        # mat = t3d.quaternions.quat2mat(env.env.get_observations()['camera_rotation'])
        # import trimesh.transformations as tra
        # tra.euler_from_matrix(mat, "rzyx")
        # np.pi/4
        # env.env.initial_joints
        print(
            "SHORT_TERM:", info["short_term_goal"], np.where(info["closest_goal_map"])
        )
        x, y = info["short_term_goal"]
        x, y = agent.semantic_map.local_to_global(x, y)
        action = ContinuousNavigationAction(np.array([x, y, 0.0]))

        # Visualize map
        depth_frame = obs.depth
        if depth_frame.max() > 0:
            depth_frame = depth_frame / depth_frame.max()
        depth_frame = (depth_frame * 255).astype(np.uint8)
        depth_frame = np.repeat(depth_frame[:, :, np.newaxis], 3, axis=2)
        vis_image = get_semantic_map_vis(
            agent.semantic_map,
            obs.task_observations["semantic_frame"],
            info["closest_goal_map"],
            depth_frame,
            env.color_palette,
            legend,
            subgoal=info["short_term_goal"],
        )
        vis_images.append(vis_image)
        cv2.imwrite(f"{output_visualization_dir}/{t}.png", vis_image[:, :, ::-1])
        cv2.imshow("vis", vis_image[:, :, ::-1])
        # map is (y,x) (bottom left of image is 0,0)

        key = cv2.waitKey(1)

        if key == ord("z"):
            break

        if key == ord("g"):
            user_input = input("Enter the goal category: ")
            print("You entered:", user_input)
            env.set_goal(user_input)

        if key == ord("t"):
            keyboard_takeover = True
            print("KEYBOARD TAKEOVER")

        # target object detected in current frame
        if (obs.semantic == env.current_goal_id).sum() > 0:
            response = obs.raw_obs["depth_response"]
            mask = obs.semantic == env.current_goal_id
            mask_points = np.stack(np.where(mask), axis=-1)
            centroid = mask_points.mean(axis=0).astype(int)
            assert mask[centroid[0], centroid[1]]
            pixel_xy = (centroid[1], centroid[0])
            point = point_from_depth_image(response, pixel_xy, BODY_FRAME_NAME)
            dist = np.linalg.norm(point)
            print("Distance to object: ", dist)
            if dist > 1.5 and dist < 3:
                print("Seeking object")
                # walk to the point as localized in the depth image
                global_point = point_from_depth_image(
                    response, pixel_xy, VISION_FRAME_NAME
                )[:2]
                cur_xy = spot.get_xy_yaw()[:2]
                delta = global_point - cur_xy
                heading = math.atan2(delta[1], delta[0])
                spot.set_base_position(
                    global_point[0],
                    global_point[1],
                    heading,
                    10,
                    relative=False,
                    max_fwd_vel=0.5,
                    max_hor_vel=0.5,
                    max_ang_vel=np.pi / 4,
                    blocking=True,
                )
        if pan_warmup:
            positions = spot.get_arm_joint_positions()
            env.env.set_arm_yaw(-np.pi, time=15)
            if positions[0] < -2.5:
                pan_warmup = False
                env.env.initialize_arm()
        else:
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
                elif key == ord("q"):
                    # rotate left
                    spot.set_base_velocity(0, 0, 0.5, 0.5)
                elif key == ord("e"):
                    # rotate right
                    spot.set_base_velocity(0, 0, -0.5, 0.5)
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
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--category", default=None)
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot, args)
