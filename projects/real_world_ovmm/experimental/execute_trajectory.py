#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Tuple

import click
import numpy as np
import rospy

from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.remote import StretchClient


def txt_to_trajectory(filename):
    """open a text file and return trajectory to execute"""
    trajectories: list[list[tuple[float, float]]] = []
    goals = []
    with open(filename, "r") as f:
        for line in f:
            if not line:
                continue
            goal, traj_str = line.strip().split("\t")
            traj_strs = [t.split(",") for t in traj_str.split(" ")]
            trajectories.append([[float(p) for p in t] for t in traj_strs])
            goals.append(goal)
    return goals, trajectories


def loose_wait(robot, x, y, theta, rate=10, pos_err_threshold=0.2, verbose=True):
    """Wait until the robot has reached a configuration... but only roughly. Used for trajectory execution."""
    rate = rospy.Rate(rate)
    xy = np.array([x, y])
    if verbose:
        print("Waiting for", x, y, theta, "threshold =", threshold)
    while not rospy.is_shutdown():
        # Loop until we get there (or time out)
        curr = robot.nav.get_base_pose()
        pos_err = np.linalg.norm(xy - curr[:2])
        if verbose:
            print("- curr pose =", curr, "target =", x, y, theta, "err =", pos_err)
        if pos_err < pos_err_threshold:
            break
        rate.sleep()


@click.command()
@click.option("--wait", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
def main(wait=False, dry_run=False):
    """Run through trajectory examples loaded in these scripts"""
    if dry_run:
        wait = False

    rospy.init_node("trajectory_execution_example")
    example_dir = "projects/real_world_ovmm/experimental"
    robot = StretchClient(init_node=False)
    robot.switch_to_navigation_mode()
    _, trajectories = txt_to_trajectory(os.path.join(example_dir, "trajectories.txt"))
    names, goal_pts = txt_to_trajectory(os.path.join(example_dir, "goals.txt"))
    for name, trajectory, goal_pt in zip(names, trajectories, goal_pts):
        print(name, goal_pt)
        print("traj:", trajectory)

    # Now start executing
    for name, trajectory, goal_pt in zip(names, trajectories, goal_pts):
        # There should only be one point in the goal point list
        goal_pt = goal_pt[0]
        # Now print info out
        print(name, goal_pt)
        # print("traj:", trajectory)
        print("Executing...")
        robot.head.look_front()
        end_idx = len(trajectory) - 1
        for i, pt in enumerate(trajectory):
            theta = 0
            x, y = pt
            if i > 0:
                pt0 = trajectory[i - 1]
                dx = x - pt0[0]
                dy = y - pt0[1]
                # print("dx, dy =", dx, dy)
                theta = np.arctan2(dy, dx)
                # print(theta)

            print(i, "=", x, y, theta)
            last_waypoint = i == end_idx
            if not dry_run:
                if last_waypoint:
                    robot.nav.navigate_to([x, y, theta], relative=False, blocking=True)
                else:
                    robot.nav.navigate_to([x, y, theta], relative=False, blocking=False)
                    loose_wait(robot, x, y, theta)

        # Now look at the goal
        dx = goal_pt[0] - x
        dy = goal_pt[1] - y
        theta = np.arctan2(dy, dx)
        if not dry_run:
            robot.nav.navigate_to([x, y, theta], relative=False, blocking=True)

        # TODO: compute head angle
        dist = np.linalg.norm([dx, dy])
        head_height = 1.41
        print("Dist to target =", dist)
        head_angle = np.tan(dist / head_height)
        head_tilt = -1 * (np.pi / 2 - head_angle)
        print("Head tilt =", head_tilt)
        if not dry_run:
            robot.head.set_pan_tilt(0, head_tilt)

        print("Looking at the", name)
        if wait:
            input("---- press enter to continue ----")
        else:
            rospy.sleep(7.5)

    robot.nav.navigate_to([0, 0, 0], relative=False, blocking=True)


if __name__ == "__main__":
    main()
