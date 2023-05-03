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
from home_robot.core.interfaces import HybridAction
from home_robot.motion.stretch import STRETCH_HOME_Q, STRETCH_PREGRASP_Q
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import (
    StretchPickandPlaceEnv,
    load_config,
)


def print_action(name, action):
    print(f"--- {name} ---")
    print("is manipulation?", action.is_manipulation())
    print("is navigation?", action.is_navigation())
    if action.is_manipulation():
        joints, xyt = action.get()
        print("action xyt =", xyt)
        print("action cfg =", joints)
    elif action.is_navigation():
        xyt = action.get()
        print("action xyt =", xyt)
    else:
        print("action sym =", action.get())


def main(**kwargs):
    config = load_config(visualize=False, **kwargs)
    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(config=config)
    obs = env.reset("table", "cup", "chair")
    robot = env.get_robot()

    action = HybridAction(robot.model.create_action_from_config(STRETCH_HOME_Q))
    print_action("HOME CONFIG", action)
    env.apply_action(action)
    input("Press enter to continue...")

    # Try a movement only action
    action = HybridAction(xyt=np.array([0, 0, 0]))
    print_action("goto(0,0,0)", action)
    env.apply_action(action)
    input("Press enter to continue...")

    # Try another test
    action = HybridAction(
        robot.model.create_action(
            lift=0.75,
            arm=0.15,
            defaults=robot.model.create_action_from_config(STRETCH_HOME_Q).joints,
        )
    )
    print_action("lift arm", action)
    env.apply_action(action)
    input("Press enter to continue...")

    pregrasp_q = robot.model.update_look_at_ee(STRETCH_PREGRASP_Q.copy())
    pregrasp_cfg = robot.model.create_action_from_config(pregrasp_q).joints
    action = HybridAction(
        robot.model.create_action(
            lift=0.4,
            arm=0.1,
            pan=0.5,
            defaults=pregrasp_cfg,
        )
    )
    print_action("down and in", action)
    env.apply_action(action)
    input("Press enter to continue...")


if __name__ == "__main__":
    main()
