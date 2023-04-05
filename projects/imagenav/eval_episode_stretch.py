#!/usr/bin/env python
import rospy
from config_utils import get_config

from home_robot.projects.imagenav.policy import CortexPolicy
from home_robot.agent.imagenav_agent.visualizer import  
from home_robot_hw.env.stretch_image_nav_env import StretchImageNavEnv

if __name__ == "__main__":
    config, config_str = get_config("configs/instance_imagenav_hm3d.yaml")

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
            high=255
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
    }

    obs_space = spaces.Dict(obs_space)

    action_space = spaces.Discrete(4)

    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchImageNavEnv(config=config)
    agent = CortexPolicy(
        config=config
        checkpoint,
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
    )

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
