# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import timeit

import numpy as np
import rospy

from home_robot.agent.motion.stretch import STRETCH_HOME_Q, HelloStretchIdx
from home_robot_hw.ros.stretch_ros import HelloStretchROSInterface

if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = HelloStretchROSInterface(
        visualize_planner=False,
    )
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q = model.update_look_at_ee(home_q.copy())
    rob.goto(q, move_base=False, wait=True)
