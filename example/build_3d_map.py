# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import timeit

import click
import numpy as np
import open3d
import rospy

from home_robot.agent.motion.stretch import HelloStretch, STRETCH_NAVIGATION_Q
from home_robot.utils.point_cloud import (
    numpy_to_pcd,
    pcd_to_numpy,
    show_point_cloud,
)
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_grasping_env import StretchGraspingEnv


def combine_point_clouds(pc_xyz: np.ndarray, pc_rgb: np.ndarray, xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Tool to combine point clouds without duplicates. Concatenate, voxelize, and then return
    the finished results."""
    if pc_rgb is None:
        pc_rgb, pc_xyz = rgb, xyz
    else:
        np.concatenate([pc_rgb, rgb], axis=0)
        np.concatenate([pc_xyz, xyz], axis=0)
    pcd = numpy_to_pcd(xyz, rgb).voxel_down_sample(voxel_size=0.01)
    return pcd_to_numpy(pcd)


class RosMapDataCollector(object):
    """Simple class to collect RGB, Depth, and Pose information for building 3d spatial-semantic
    maps for the robot. Needs to subscribe to:
    - color images
    - depth images
    - camera info
    - joint states/head camera pose
    - base pose (relative to world frame)

    This is an example collecting the data; not necessarily the way you should do it.
    """

    def __init__(self, env, visualize_planner=False):
        self.env = env  # Get the connection to the ROS environment via agent
        self.observations = []
        self.started = False
        self.robot_model = HelloStretch(visualize=visualize_planner)

    def step(self):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        rgb, depth, xyz = self.env.get_images(compute_xyz=True)
        q, dq = self.env.update()

        # apply depth filter
        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        xyz = xyz.reshape(-1, 3)
        valid_depth = np.bitwise_and(depth > 0.1, depth < 4.)
        rgb = rgb[valid_depth, :]
        xyz = xyz[valid_depth, :]
        self.observations.append((rgb, xyz, q, dq))

    def show(self):
        """Display the aggregated point cloud."""

        # Create a combined point cloud
        # Do the other stuff we need
        pc_xyz, pc_rgb = None, None
        for obs in self.observations:
            rgb = obs[0]
            xyz = obs[1]
            pc_xyz, pc_rgb = combine_point_clouds(pc_xyz, pc_rgb, xyz, rgb)

        # Visualize point clloud + origin
        show_point_cloud(pc_xyz, pc_rgb / 255, orig=np.zeros(3))


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--max-frames", default=50, type=int)
@click.option("--visualize", default=False, is_flag=True)
def main(rate, max_frames, visualize):
    rospy.init_node("build_3d_map")
    env = StretchGraspingEnv(segmentation_method=None)
    collector = RosMapDataCollector(env, visualize)

    # Tuck the arm away
    env.goto(STRETCH_NAVIGATION_Q, wait=False)

    rate = rospy.Rate(rate)

    # Move the robot
    # TODO: replace env with client
    if not env.in_navigation_mode():
        env.switch_to_navigation_mode()
    # Sequence information if we are executing the ttrajectory
    step = 0
    # Number of frames collected
    frames = 0

    print("Press ctrl+c to finish...")
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        # Run until we control+C this script
        collector.step()  # Append latest observations

        ti = (rospy.Time.now() - t0).to_sec()
        print("Time =", ti)
        if step == 0 and ti < 5.0:
            env.navigate_to((0.5, 0, 0))
        elif ti >= 5.0 and step <= 1 and ti < 15.:
            env.navigate_to((0.5, 0.5, np.pi / 2))
            step = 1
        elif ti >= 15. and step <= 2:
            env.navigate_to((0, 0, 0))
            step = 2

        frames += 1
        if max_frames > 0 and frames >= max_frames:
            break

        rate.sleep()

    print("Done collecting data.")
    env.navigate_to((0, 0, 0))
    collector.show()


if __name__ == "__main__":
    """run the test script."""
    main()
