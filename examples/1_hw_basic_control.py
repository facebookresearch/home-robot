# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchKinematics
from home_robot_hw.remote import StretchClient

if __name__ == "__main__":
    robot = StretchClient()
    model = HelloStretchKinematics()

    # Acquire camera observations
    imgs = robot.head.get_images()

    # Get camera pose
    camera_pose = robot.head.get_pose()
    print(f"camera_pose={camera_pose}")

    # Move camera
    robot.head.set_pan_tilt(pan=np.pi / 4, tilt=-np.pi / 3)
    robot.head.look_at_ee()
    robot.head.look_ahead()

    # Switch to navigation mode
    robot.switch_to_navigation_mode()
    assert robot.in_navigation_mode()

    # Get base pose
    xyt = robot.nav.get_base_pose()
    print(f"Base pose: xyt={xyt}")

    # Command the robot to navigate to a waypoint
    xyt_goal = [0.15, 0.15, -np.pi / 4]
    robot.nav.navigate_to(xyt_goal)

    # Home robot base (navigate back to origin)
    robot.nav.home()

    # Switch to manipulation mode
    robot.switch_to_manipulation_mode()
    assert robot.in_manipulation_mode()

    # Home robot joints (moves to predefined home joint configuration)
    robot.manip.home()

    # Get gripper pose
    pos, quat = robot.manip.get_ee_pose()
    print(f"EE pose: pos={pos}, quat={quat}")

    # Command the robot arm 1: Direct joint control
    # (joints: [base translation, arm lift, arm extend, gripper yaw, gripper pitch, gripper roll])
    q_desired = np.array([-0.1, 0.5, 0.3, 0, 0, 0])
    robot.manip.goto_joint_positions(q_desired)

    # Command the robot arm 2: Absolute EE control
    pos_desired = np.array([0.1, -0.2, 0.4])
    quat_desired = np.array([-0.7079143, 0.12421559, 0.1409881, -0.68084526])
    robot.manip.goto_ee_pose(pos_desired, quat_desired, relative=False)

    # Command the robot arm 3: Relative EE control
    #   (note: orientation stays the same if not specified)
    pos_desired = np.array([-0.1, -0.1, -0.1])
    robot.manip.goto_ee_pose(pos_desired, relative=True)

    # Command the robot arm 4: Simple EE rotations
    #   (rotates around Z axis by 0.5 radians)
    robot.manip.rotate_ee(axis=2, angle=0.5)

    # Command the robot arm 5: For backward compatibility
    robot.manip.goto(STRETCH_HOME_Q)

    # Gripper commands
    robot.manip.open_gripper(blocking=True)
    robot.manip.close_gripper()

    # Test command in wrong mode
    assert robot.in_manipulation_mode()
    try:
        robot.nav.navigate_to()
    except TypeError:
        pass  # prints out an rospy.logerr that alerts the user of erroneous mode

    # Some commands are still available
    xyt = robot.nav.get_base_pose()
    print(f"Base pose: xyt={xyt}")

    # Stop all robot motion
    robot.stop()
