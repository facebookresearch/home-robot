import os

import numpy as np
import yaml

from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchKinematics
from home_robot_hw.remote import StretchClient

if __name__ == "__main__":
    # load action dict from "~/actions.yaml"
    actions = yaml.unsafe_load(open(os.path.expanduser("~/actions.yaml"), "r"))
    robot = StretchClient()
    model = HelloStretchKinematics(ik_type="pinocchio")

    # Switch to manipulation mode
    robot.switch_to_manipulation_mode()
    assert robot.in_manipulation_mode()

    # Home robot joints (moves to predefined home joint configuration)
    # robot.manip.home()
    # robot.manip.open_gripper(blocking=True)

    # Get gripper pose
    pos, quat = robot.manip.get_ee_pose()
    print(f"EE pose: pos={pos}, quat={quat}")
    base_pose_start = robot.nav.get_base_pose()

    # Command the robot arm 2: Absolute EE control
    move_base_flag = True
    for i, action in enumerate(actions):
        base_pose = robot.nav.get_base_pose()
        print(base_pose)
        input("Press Enter")
        pos_desired = action["pos"]
        quat_desired = action["ori"]
        q0 = robot.manip.get_joint_positions()
        if i > 0:
            pos_desired[0] = 0.0
            # move_base_flag = False
            action["gripper"] = 1
        print(f"EE pose: pos={pos_desired}, quat={quat_desired}")
        robot.manip.goto_ee_pose(
            pos_desired,
            quat_desired,
            relative=False,
            world_frame=True,
            initial_cfg=q0,
            # move_base=move_base_flag,
        )
        if action["gripper"] == 1:
            robot.manip.close_gripper(blocking=True)
        else:
            robot.manip.open_gripper(blocking=True)
        input("Press enter to continue")

    # # Command the robot arm 3: Relative EE control
    # #   (note: orientation stays the same if not specified)
    # pos_desired = np.array([-0.1, -0.1, -0.1])
    # robot.manip.goto_ee_pose(pos_desired, relative=True)

    # Stop all robot motion
    robot.stop()
