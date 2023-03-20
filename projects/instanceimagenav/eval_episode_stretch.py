#!/usr/bin/env python
import rospy
from config_utils import get_config

from home_robot.agent.imagenav_agent.imagenav_agent import ImageNavAgent
from home_robot.agent.imagenav_agent.visualizer import record_video
from home_robot_hw.env.stretch_image_nav_env import StretchImageNavEnv

if __name__ == "__main__":
    config, config_str = get_config("configs/instance_imagenav_hm3d.yaml")

    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchImageNavEnv(config=config)
    agent = ImageNavAgent(config=config)

    print()
    print("==============")
    env.switch_to_navigation_mode()
    print("==============")

    env.reset()
    agent.reset()

    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        action = agent.act(obs)
        env.apply_action(action)

    record_video(
        target_dir=f"{config.dump_location}/videos/{config.exp_name}",
        image_dir=f"{config.dump_location}/images/{config.exp_name}",
    )
