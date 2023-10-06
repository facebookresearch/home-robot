# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import timeit
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch

import home_robot.utils.depth as du
from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from home_robot.mapping.voxel import plan_to_frontier

# Import planning tools for exploration
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.perception.encoders import ClipEncoder
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
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

    def __init__(
        self,
        robot,
        semantic_sensor=None,
        visualize_planner=False,
        voxel_size: float = 0.01,
        encoder: Optional[ClipEncoder] = None,
        **kwargs,
    ):
        self.robot = robot  # Get the connection to the ROS environment via agent
        # Run detection here
        self.semantic_sensor = semantic_sensor
        self.encoder = encoder
        self.started = False
        self.robot_model = HelloStretchKinematics(visualize=visualize_planner)
        self.voxel_map = SparseVoxelMap(
            resolution=voxel_size, encoder=self.encoder, **kwargs
        )

    def get_planning_space(self) -> SparseVoxelMapNavigationSpace:
        """return space for motion planning. Hard codes some parameters for Stretch"""
        return SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot_model,
            step_size=0.1,
            dilate_frontier_size=12,  # 0.6 meters back from every edge
            dilate_obstacle_size=5,
        )

    def step(self, visualize_map=False):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        obs = self.robot.get_observation()

        # Semantic prediction
        obs = self.semantic_sensor.predict(obs)

        # Add observation - helper function will unpack it
        self.voxel_map.add_obs(
            obs,
            K=torch.from_numpy(self.robot.head._ros_client.rgb_cam.K).float(),
        )
        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

    def get_2d_map(self):
        """Get 2d obstacle map for low level motion planning and frontier-based exploration"""
        return self.voxel_map.get_2d_map()

    def get_xyz_rgb(self):
        points, _, _, rgb = self.voxel_map.voxel_pcd.get_pointcloud()
        return points, rgb

    def show(self, orig: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        return self.voxel_map.show(orig=orig)
