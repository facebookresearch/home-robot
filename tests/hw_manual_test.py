# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import rospy

from home_robot.motion.stretch import HelloStretch, STRETCH_HOME_Q
from home_robot_hw.env.stretch_abstract_env import StretchEnv

# Loose tolerances just to test that the robot moved reasonably
POS_TOL = 0.1
YAW_TOL = 0.2

class StretchEnvImpl(StretchEnv):
    def reset(self):
        pass

    def apply_action(self, action, info = None):
        pass

    def get_observation(self):
        pass

    @property
    def episode_over(self):
        pass

    def get_episode_metrics(self):
        pass

if __name__ == '__main__':
    print("Initializing...")
    rospy.init_node("hw_manual_test")
    robot = StretchEnvImpl()

    model = HelloStretch()

    # Home robot
    print("Homing robot...")
    q = STRETCH_HOME_Q
    robot.pretty_print(q)
    robot.goto(q)

    # Head movement
    print("Testing robot joint movement...")

    q = model.update_look_at_ee(q)
    robot.goto(q, wait=True)
    q = model.update_look_front(q)
    robot.goto(q, wait=True)

    print(f"Confirm that the robot head has moved accordingly.")
    input("(press enter to continue)")

    # Navigation
    print("Testing robot navigation...")
    robot.switch_to_navigation_mode()

    xyt_goal = [0.25, 0.25, -np.pi/2]
    robot.navigate_to(xyt_goal, blocking=True)
    
    xyt_curr = robot.get_base_pose()
    assert np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL)
    assert np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL)

    print(f"Confirm that the robot moved to {xyt_goal} (forward left, facing right)")
    input("(press enter to continue)")

    # Manipulation
    print("Testing robot manipulation...")
    robot.switch_to_manipulation_mode()

    pos_diff_goal = np.array([0.1, 0.0, 0.1])
    pos, quat = model.get_ee_pose(q)
    q = model.manip_ik((pos + pos_diff_goal, quat))
    robot.goto(q, wait=True)

    print(f"Confirm that the robot EE moved by {pos_diff_goal} (lift upwards and extend outwards by 10cm)")
    input("(press enter to continue)")