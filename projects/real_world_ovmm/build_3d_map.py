# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import timeit
from typing import Tuple

import click
import numpy as np
import open3d
import rospy
import trimesh
import trimesh.transformations as tra

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion.stretch import STRETCH_NAVIGATION_Q, HelloStretchKinematics
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.remote import StretchClient


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

    def __init__(self, robot, visualize_planner=False):
        self.robot = robot  # Get the connection to the ROS environment via agent
        self.started = False
        self.robot_model = HelloStretchKinematics(visualize=visualize_planner)
        self.voxel_map = SparseVoxelMap(resolution=0.01)

    def step(self):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        rgb, depth, xyz = self.robot.head.get_images(
            compute_xyz=True,
        )
        # Get the camera pose and make sure this works properly
        camera_pose = self.robot.head.get_pose(rotated=False)
        # Get RGB and depth as necessary
        orig_rgb = rgb.copy()
        orig_depth = depth.copy()

        # apply depth filter
        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        xyz = xyz.reshape(-1, 3)
        valid_depth = np.bitwise_and(depth > 0.1, depth < 4.0)
        rgb = rgb[valid_depth, :]
        xyz = xyz[valid_depth, :]
        # TODO: remove debug code
        # For now you can use this to visualize a single frame
        # show_point_cloud(xyz, rgb / 255, orig=np.zeros(3))
        self.voxel_map.add(
            camera_pose,
            xyz,
            rgb,
            depth=depth,
            K=self.robot.head._ros_client.rgb_cam.K,
            orig_rgb=orig_rgb,
            orig_depth=orig_depth,
        )

    def show(self) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""

        # Create a combined point cloud
        # Do the other stuff we need
        pc_xyz, pc_rgb = self.voxel_map.get_data()
        show_point_cloud(pc_xyz, pc_rgb / 255, orig=np.zeros(3))
        return pc_xyz, pc_rgb


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--max-frames", default=20, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--pcd-filename", default="output.ply", type=str)
@click.option("--pkl-filename", default="output.pkl", type=str)
def main(rate, max_frames, visualize, manual_wait, pcd_filename, pkl_filename):
    robot = StretchClient()
    collector = RosMapDataCollector(robot, visualize)

    # Tuck the arm away
    print("Sending arm to  home...")
    robot.switch_to_manipulation_mode()

    robot.head.look_front(blocking=False)
    robot.manip.goto_joint_positions(
        robot.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
    )
    print("... done.")

    rate = rospy.Rate(rate)

    # Move the robot
    robot.switch_to_navigation_mode()
    # Sequence information if we are executing the trajectory
    step = 0
    # Number of frames collected
    frames = 0

    trajectory = [
        (0, 0, 0),
        (0.4, 0, 0),
        (0.75, 0.15, np.pi / 4),
        (0.85, 0.3, np.pi / 4),
        (0.95, 0.5, np.pi / 2),
        (1.0, 0.55, np.pi),
        (0.6, 0.45, 9 * np.pi / 8),
        (0.0, 0.3, -np.pi / 2),
        (0, 0, 0),
        (0.2, 0, 0),
        (0.5, 0, 0),
        (0.7, 0.2, np.pi / 4),
        (0.7, 0.4, np.pi / 2),
        (0.5, 0.4, np.pi),
        (0.2, 0.2, -np.pi / 4),
        (0, 0, -np.pi / 2),
        (0, 0, 0),
    ]

    collector.step()  # Append latest observations
    # print("Press ctrl+c to finish...")
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        # Run until we control+C this script

        ti = (rospy.Time.now() - t0).to_sec()
        print("t =", ti, trajectory[step])
        robot.nav.navigate_to(trajectory[step])
        print("... done navigating.")
        if manual_wait:
            input("... press enter ...")
        print("... capturing frame!")
        step += 1

        collector.step()  # Append latest observations

        frames += 1
        if max_frames > 0 and frames >= max_frames or step >= len(trajectory):
            break

        rate.sleep()

    print("Done collecting data.")
    robot.nav.navigate_to((0, 0, 0))
    pc_xyz, pc_rgb = collector.show()

    # Create pointcloud
    if len(pcd_filename) > 0:
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(pcd_filename, pcd)
    if len(pkl_filename) > 0:
        collector.voxel_map.write_to_pickle(pkl_filename)

    rospy.signal_shutdown("done")


if __name__ == "__main__":
    """run the test script."""
    main()
