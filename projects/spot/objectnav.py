import pickle
import sys
from pathlib import Path
from typing import List
import math
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
import time

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from spot_wrapper.spot import Spot

import home_robot.utils.visualization as vu
import home_robot.utils.pose as pu
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import DiscreteNavigationAction
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
    SEM_START = 6

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
            subgoal_map[subgoal[0],subgoal[1]] = 1
            subgoal_mat = (1 - skimage.morphology.binary_dilation(subgoal_map, selem)) != 1
            subgoal_mask = subgoal_mat == 1
            # hack for now
            semantic_categories_map[subgoal_mask] = PI.REST_OF_GOAL

    # Draw semantic map
    semantic_map_vis = Image.new("P", semantic_categories_map.shape)
    semantic_map_vis.putpalette(map_color_palette)
    semantic_map_vis.putdata(semantic_categories_map.flatten().astype(np.uint8))
    semantic_map_vis = semantic_map_vis.convert("RGB")
    semantic_map_vis = np.flipud(semantic_map_vis)
    semantic_map_vis = cv2.resize(semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
)
    # from matplotlib import pyplot as plt
    # breakpoint()
    # plt.imshow(semantic_map_vis)
    # plt.imshow(subgoal_map)
    # plt.show()
    
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


def main(spot):
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

    env = SpotObjectNavEnv(spot,position_control=True)
    env.reset()
    env.set_goal("oven")

    agent = ObjectNavAgent(config=config)
    agent.reset()

    assert agent.num_sem_categories == env.num_sem_categories
    pan_warmup = False
    keyboard_takeover = False
    if pan_warmup:
        positions = spot.get_arm_joint_positions()
        new_pos = positions.copy()
        new_pos[0] = np.pi
        spot.set_arm_joint_positions(new_pos,travel_time=3)
        time.sleep(3)
    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)

        obs = env.get_observation()
        with open(f"{obs_dir}/{t}.pkl", "wb") as f:
            pickle.dump(obs, f)

        action, info = agent.act(obs)
        print("SHORT_TERM:", info['short_term_goal'])
        x,y = info['short_term_goal']
        lmb = agent.semantic_map.get_planner_pose_inputs(0)[3:]
        x=x-240 + (lmb[0] - 240)
        y=y-240 + (lmb[2] - 240)
        # angle from the origin to the STG
        angle_st_goal = math.atan2(x, y)
        dist = np.linalg.norm((x,y))*0.05
        xg = dist*np.cos(angle_st_goal + env.start_compass) + env.start_gps[0]
        yg = dist*np.sin(angle_st_goal + env.start_compass) + env.start_gps[1]
        
        # compute the angle from the current pose to the destination point
        # in robot global frame
        cx,cy,yaw = spot.get_xy_yaw()
        angle = math.atan2((yg-cy),(xg-cx)) % (2*np.pi)
        # angle
        # angle
        # angle_st_goal + env.start_compass
        
        # 
        # dist = np.linalg.norm(stg[:2])
        # angle = -stg[2]*np.pi/180
        # pos = np.array([np.cos(angle),np.sin(angle)])*dist*0.05
        # pos = np.array(stg[:2])*0.05
        # pos = np.array((stg[1],stg[0]))*0.05

        # x,y,yaw = spot.get_xy_yaw()
        # relative_angle = angle_st_goal + env.start_compass - yaw
        # relative_angle = pu.normalize_radians(relative_angle)
        # print("Relative Angle: ",relative_angle)
        # if abs(relative_angle) > np.pi/6:
        #     # while abs(relative_angle) > np.pi/2:
        #     x,y,yaw = spot.get_xy_yaw()
        #     relative_angle = angle_st_goal + env.start_compass - yaw
        #     relative_angle = pu.normalize_radians(relative_angle)
        #     vel = 0.5
        #     rot = -vel if relative_angle < 0 else vel
        #     spot.set_base_velocity(0,0,rot,1)  
        #     print("Angle too steep, rotating by ", rot)
        #     action = None  
        # else:
        # action = [xg,yg,angle_st_goal+env.start_compass]
        action = [xg,yg,angle]
        print("ObjectNavAgent point action", action)
        # diff = action - np.array(spot.get_xy_yaw())
        # diff[2] % 2*np.pi
        # print("ObjectNavAgent action", action)

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
            subgoal = info['short_term_goal']
        )
        vis_images.append(vis_image)
        cv2.imshow("vis", vis_image[:, :, ::-1])
        key = cv2.waitKey(50)
        if key == ord('z'):
            break
        if key == ord('q'):
            keyboard_takeover = True
            print("KEYBOARD TAKEOVER")
        if keyboard_takeover:
            if key == ord("w"):
                spot.set_base_velocity(0.5,0,0,0.5)
            elif key == ord("s"):
                # back
                spot.set_base_velocity(-0.5,0,0,0.5)
            elif key == ord("a"):
                # left
                spot.set_base_velocity(0,0.5,0,0.5)
            elif key == ord("d"):
                # right
                spot.set_base_velocity(0,-0.5,0,0.5)
        else:
            if pan_warmup:
                positions = spot.get_arm_joint_positions()
                new_pos = positions.copy()
                new_pos[0] = -np.pi
                spot.set_arm_joint_positions(new_pos,travel_time=20)
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
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
