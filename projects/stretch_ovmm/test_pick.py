#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This file is a test for inverse kinematics on the stretch robot. It makes sure we can reach and execute different positions, which have been generated by the grasp planner. It's a useful utility for now.
"""

import numpy as np
import rospy

from home_robot.agent.hierarchical.pick_and_place_agent import PickAndPlaceAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


def run_experiment():
    config_path = "projects/stretch_grasping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(config=config)
    env.reset("table", "cup", "chair")

    # pose = np.array(
    #    [
    #        [0.23301425, -0.97144842, -0.04463536, -0.00326367],
    #        [-0.97188458, -0.23103087, -0.04544342, -0.44448592],
    #        [0.0338338, 0.05396939, -0.99796923, 0.99206106],
    #        [
    #            0.0,
    #            0.0,
    #            0.0,
    #            1.0,
    #        ],
    #    ]
    # )
    pose = np.array(
        [
            [0.63598766, 0.75559136, 0.15684828, -0.12786708],
            [0.70388954, -0.65130729, 0.28344016, -0.57091706],
            [0.31632136, -0.06986059, -0.94607626, 1.01114796],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # - with theta x/y from vertical = 0.16429382745460722 0.2910
    pos, quat = to_pos_quat(pose)

    # Debugging
    pose1 = env.robot.head.get_pose()
    pose2 = env.robot.head.get_pose_in_base_coords()
    print("head pose in world coords:")
    print(pose1)
    print("head pose in base coords:")
    print(pose2)

    env.grasp_planner.go_to_manip_mode()
    env.grasp_planner.try_executing_grasp(pose, wait_for_input=True)


if __name__ == "__main__":
    run_experiment()
