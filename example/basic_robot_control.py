import time

import numpy as np

from home_robot.motion.stretch import HelloStretch
from home_robot_hw.client import StretchClient

if __name__ == "__main__":
    robot = StretchClient()
    model = HelloStretch()

    # Acquire camera observations
    imgs = robot.camera.get_images()

    # Get camera pose
    camera_pose = robot.camera.get_pose()

    # Move camera
    robot.camera.set_pan_tilt(pan=np.pi / 4, tilt=-np.pi / 3)

    # Switch to navigation mode
    robot.switch_to_navigation_mode()
    assert robot.in_navigation_mode()

    # Get base pose
    xyt = robot.nav.get_pose()

    # Command robot velocities
    robot.nav.set_velocity(v=0.1, w=0.0)
    time.sleep(1)
    robot.nav.set_velocity(v=0.0, w=0.0)

    # Command the robot to navigate to a waypoint
    xyt_goal = [0.25, 0.25, -np.pi / 2]
    robot.nav.navigate_to(xyt_goal, blocking=True)

    # Switch to manipulation mode
    robot.switch_to_manipulation_mode()
    assert robot.in_manipulation_mode()

    # Get ee pose
    pos, quat = robot.manip.get_ee_pose(relative=True)

    # Command the robot arm 1
    q_desired = np.random.randn(6)
    pos_desired = np.array([-0.10281811, -0.7189281, 0.71703106])
    quat_desired = np.array([-0.7079143, 0.12421559, 0.1409881, -0.68084526])

    robot.manip.set_joint_positions(q_desired, blocking=True)
    robot.manip.set_ee_pose(pos_desired, quat_desired, blocking=True)

    # Command the robot arm 2
    q = model.compute_ik(pos_desired, quat_desired)
    robot.manip.goto(q)

    # Test command in wrong mode
    assert robot.in_manipulation_mode()
    try:
        robot.nav.navigate_to()
    except TypeError:
        pass  # prints out an rospy.logerr that alerts the user of erroneous mode

    # Some commands are still available
    xyt = robot.nav.get_pose()

    # Stop all robot motion
    robot.stop()
