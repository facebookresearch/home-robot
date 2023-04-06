#!/usr/bin/env python
import numpy as np
import rospy

from home_robot.agent.hierarchical.pick_and_place_agent import PickAndPlaceAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv

if __name__ == "__main__":
    config_path = "projects/stretch_grasping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    agent = PickAndPlaceAgent(config=config)
    env = StretchPickandPlaceEnv(config=config)

    agent.reset()
    env.reset("cup")

    env.navigate_to([0, 0, np.pi / 2], relative=True, blocking=True)
