#!/usr/bin/env python
import os
import time
import numpy as np
from PIL import Image

from gym import spaces
from gym.spaces import Dict as SpaceDict
import rospy
import torch

from policy import CortexPolicy

import hydra
from omegaconf import OmegaConf

from habitat.config.default import Config as CN
from habitat_vc.config import get_config

from home_robot.agent.imagenav_agent.visualizer import record_video
from home_robot.motion.stretch import STRETCH_NAVIGATION_Q
from home_robot_hw.env.stretch_old_image_nav_env import StretchImageNavEnv

from home_robot.core.interfaces import DiscreteNavigationAction

@hydra.main(config_path="configs", config_name="config_imagenav_stretch")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = CN(cfg)

    config = get_config()
    config.merge_from_other_cfg(cfg)

    print("Load Config")

    h, w = (
        640,
        480,
    )

    obs_space = {
        "rgb": spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
        "imagegoalrotation": spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
    }

    obs_space = spaces.Dict(obs_space)

    action_space = spaces.Discrete(4)

    print("define action and obs space")

    rospy.init_node("eval_episode_stretch_objectnav")

    print("init rospy")

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location="cpu"
    )

    env = StretchImageNavEnv(config=config)
    agent = CortexPolicy(
        config=config,
        checkpoint=checkpoint,
        observation_space=obs_space,
        action_space=action_space,
        device="cuda",
    )

    print()
    print("==============")
    env.robot.switch_to_manipulation_mode()
    env.robot.manip.goto_joint_positions(
        env.robot.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
    )
    env.robot.switch_to_navigation_mode()
    env.robot.head.set_pan_tilt(tilt=config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION[0], blocking=True)
    print("==============")

    env.reset()
    agent.reset()
    env.robot.head.set_pan_tilt(tilt=config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION[0], blocking=True)    

    # create folders for saving images
    folder_name = time.strftime("%d-%m-%Y_%H:%M:%S", time.localtime())
    folder_name = f"videos/{folder_name}"
    images_folder = f"{folder_name}/images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    action = None
    t = 0
    time_secs = time.time()
    try:
        while (action != DiscreteNavigationAction.STOP) and t < 500:
            t += 1
            print("STEP =", t)
            obs = env.get_observation()
            
            rgb_image = Image.fromarray(obs['rgb'])
            rgb_image.save('{}/snapshot_{}.png'.format(images_folder, t))

            if t == 1:
                goal_image = Image.fromarray(obs['imagegoalrotation'])
                goal_image.save('{}/goal_image.png'.format(images_folder))


            action = agent.act(obs)
            env.apply_action(action)
            print("Time: {} | Action: {}".format(t, action))
    except:
        print("Saving Video")
    
    print("Total Time: {}".format(time.time() - time_secs))

    record_video(
        target_dir=folder_name,
        image_dir=images_folder,
        fps=5
    )

if __name__ == "__main__":
    main()
