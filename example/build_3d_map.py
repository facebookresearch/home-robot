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

from home_robot.agent.motion.stretch import STRETCH_PREGRASP_Q, HelloStretchIdx
from home_robot.utils.data_tools.point_cloud import (
    numpy_to_pcd,
    pcd_to_numpy,
    show_point_cloud,
)
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_grasping_env import StretchGraspingEnv


def combine_point_clouds(pc_xyz, pc_rgb, xyz, rgb):
    """Tool to combine point clouds without duplicates. Concatenate, voxelize, and then return
    the finished results."""
    if pc_rgb is None:
        pc_rgb, pc_xyz = rgb, xyz
    else:
        np.concatenate([pc_rgb, rgb], axis=0)
        np.concatenate([pc_xyz, xyz], axis=0)
    pcd = numpy_to_pcd(xyz, rgb).voxel_down_sample(voxel_size=0.01)
    open3d.visualization.draw_geometries([pcd])
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

    def __init__(self, env):
        self.env = env  # Get the connection to the ROS environment via agent
        self.observations = []
        self.started = False

    def step(self):
        """Step the collector. Get a single observation of the world."""
        rgb, depth, xyz = self.env.get_images(compute_xyz=True)
        q, dq = self.env.update()
        self.observations.append((rgb, depth, xyz, q, dq))

    def show(self):
        """Display the aggregated point cloud."""

        # Create a combined point cloud
        # Do the other stuff we need
        pc_xyz, pc_rgb = None, None
        for obs in self.observations:
            rgb = obs[0].reshape(-1, 3)
            # dpt = obs[1].reshape(-1)
            xyz = obs[2].reshape(-1, 3)
            pc_xyz, pc_rgb = combine_point_clouds(pc_xyz, pc_rgb, xyz, rgb)
            print(pc_xyz.shape, pc_rgb.shape)


@click.command()
@click.option("--rate", default=10, type=int)
def main(rate=10):
    rospy.init_node("build_3d_map")
    env = StretchGraspingEnv(segmentation_method=None)
    collector = RosMapDataCollector(env)

    rate = rospy.Rate(rate)
    print("Press ctrl+c to finish...")
    while not rospy.is_shutdown():
        # Run until we control+C this script
        collector.step()  # Append latest observations
        rate.sleep()
        # TODO: remove debug break
        break

    print("Done collecting data.")
    collector.show()


if __name__ == "__main__":
    """run the test script."""
    main()
