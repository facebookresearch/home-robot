# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from home_robot_hw.remote import StretchClient

# Loose tolerances just to test that the robot moved reasonably
POS_TOL = 0.1
YAW_TOL = 0.2


if __name__ == "__main__":
    print("Initializing...")
    robot = StretchClient()

    # Reset robot
    print("Resetting robot...")
    robot.reset()

    # Head movement
    print("Testing robot head movement...")
    robot.head.look_at_ee()
    robot.head.look_front()

    print("Confirm that the robot head has moved accordingly.")
    input("(press enter to continue)")

    # Navigation
    print("Testing robot navigation...")
    robot.switch_to_navigation_mode()

    xyt_goal = [0.25, 0.25, -np.pi / 2]
    robot.nav.navigate_to(xyt_goal)

    # Sometimes the robot will time out before done moving.
    print("Make sure orientation is correct...")
    for _ in range(3):
        print("- sending the command again to guard against timeouts...")
        robot.nav.navigate_to(xyt_goal)
    xyt_curr = robot.nav.get_base_pose()

    print("Current orientation:", xyt_curr[2], "goal was", xyt_goal[2])
    assert np.allclose(xyt_curr[2], xyt_goal[2], atol=YAW_TOL)
    print("Current location:", xyt_curr[:2], "goal was", xyt_goal[:2])
    assert np.allclose(xyt_curr[:2], xyt_goal[:2], atol=POS_TOL)

    print(f"Confirm that the robot moved to {xyt_goal} (forward left, facing right)")
    input("(press enter to continue)")

    # Manipulation
    print("Testing robot manipulation...")
    robot.switch_to_manipulation_mode()

    pos_diff_goal = np.array([0.2, -0.2, 0.2])
    robot.manip.goto_ee_pose(pos_diff_goal, relative=True)

    print(
        f"Confirm that the robot EE moved by {pos_diff_goal} (+X is forward, +Y is left, +Z is up)"
    )
    input("(press enter to continue)")

    print("Test complete!")
