# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Iterable, List, Optional, Tuple

import click
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from home_robot.manipulation.voxel_grasps import VoxelGraspGenerator
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.ros.grasp_helper import GraspServer


@click.command()
@click.option("--debug", default=False, is_flag=True)
def inference(debug):
    """
    Predict 6-DoF grasp distribution for given point cloud with a heuristic

    Heuristic is as follows:
        1. Voxelize the point cloud
        2. Extract the top 10% voxels with highest Z coordinates in world frame
        3. Project the said voxels into a 2D occupancy map
        4. Compute grasp scores for each voxel based on neighboring occupancies
        5. Generate top-down grasps if score > threshold
    """

    in_base_frame = True  # produced grasps are in base frame
    grasp_generator = VoxelGraspGenerator(in_base_frame, debug)

    # Initialize server
    _ = GraspServer(grasp_generator.get_grasps)

    # Spin
    rospy.spin()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        continue


if __name__ == "__main__":
    rospy.init_node("simple_grasp_server")
    inference()
