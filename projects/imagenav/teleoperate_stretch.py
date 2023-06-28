#!/usr/bin/env python
import numpy as np
from PIL import Image

import rospy

import hydra
from omegaconf import OmegaConf

from habitat.config.default import Config as CN
from habitat_vc.config import get_config

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

    print("define action and obs space")

    rospy.init_node("eval_episode_stretch_objectnav")

    print("init rospy")

    env = StretchImageNavEnv(config=config)

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
    env.robot.head.set_pan_tilt(tilt=config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION[0], blocking=True)    

    # create folders for saving images
    folder_name = "/home/cortex/home-robot/projects/imagenav/image_goals/final_goals"

    
    action = None
    t = 0
    try:
        while (action != DiscreteNavigationAction.STOP) and t < 500:
            t += 1
            
            obs = env.get_observation()
            rgb_image = Image.fromarray(obs['rgb'])
            rgb_image.show("Current RGB Image")

            key_pressed = input("Enter command: ")

            if key_pressed == 'w' or key_pressed == 'W':
                action = DiscreteNavigationAction.MOVE_FORWARD
                env.apply_action(action)
            elif key_pressed == 'a' or key_pressed == 'A':
                action = DiscreteNavigationAction.TURN_LEFT
                env.apply_action(action)
            elif key_pressed == 'd' or key_pressed == 'D':
                action = DiscreteNavigationAction.TURN_RIGHT
                env.apply_action(action)
            elif key_pressed == 's' or key_pressed == 'S':
                action = DiscreteNavigationAction.STOP
                env.apply_action(action)
                exit(0)
            elif key_pressed == 'b' or key_pressed == 'B':
                start_name = input('Enter Episode Number: ')
                rgb_image.save('{}/start_images/start_image_{}.png'.format(folder_name, start_name))
            elif key_pressed == 'g' or key_pressed == 'G':
                goal_name = input('Enter Episode Number: ')
                rgb_image.save('{}/goal_image_{}.png'.format(folder_name, goal_name))


            print("Time: {} | Action: {}".format(t, action))
    except:
        env.robot.switch_to_navigation_mode()


if __name__ == "__main__":
    main()
