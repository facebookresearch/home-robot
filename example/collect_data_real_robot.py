import argparse
import rospy
import timeit
import numpy as np

from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot.agent.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.agent.perception.constants import coco_categories
from home_robot.utils.pose import to_pos_quat

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("collect_data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    args = parse_args()
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=False)

    # Get cameras from the robot object
    t0 = timeit.default_timer()
    rgb_cam = rob.rgb_cam
    dpt_cam = rob.dpt_cam
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()
    print("took", timeit.default_timer() - t0, "seconds to get images")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q, _ = rob.update()
    print("q =", q)
    q = model.update_look_at_ee(home_q.copy())
    q = model.update_gripper(q, open=True)
    rob.goto(q, move_base=False, wait=True)
    model = rob.get_model()
    q, _ = rob.update()
    print("q =", q)

    input("Press enter when ready to collect frames")
    rate = rospy.rate(10)
    for i in range(100):
        # add frames and joint state info to buffer
        q, dq = rob.update()
        rate.sleep()
