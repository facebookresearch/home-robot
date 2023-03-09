# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import numpy as np
import trimesh

from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy


def combine_point_clouds(
    pc_xyz: np.ndarray,
    pc_rgb: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    sparse_voxel_size: float = 0.01,
) -> np.ndarray:
    """Tool to combine point clouds without duplicates. Concatenate, voxelize, and then return
    the finished results."""
    if pc_rgb is None:
        pc_rgb, pc_xyz = rgb, xyz
    else:
        pc_rgb = np.concatenate([pc_rgb, rgb], axis=0)
        pc_xyz = np.concatenate([pc_xyz, xyz], axis=0)
    pcd = numpy_to_pcd(pc_xyz, pc_rgb).voxel_down_sample(voxel_size=sparse_voxel_size)
    return pcd_to_numpy(pcd)


class SparseVoxelMap(object):
    """Create a voxel map object which captures 3d information."""

    def __init__(self, resolution=0.01, feature_dim=3):
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.xyz = None
        self.feats = None
        self.observations = []

    def add(self, camera_pose: np.ndarray, xyz: np.ndarray, feats: np.ndarray):
        """Add this to our history of observations. Also update the current running map."""
        assert xyz.shape[-1] == 3
        assert feats.shape[-1] == self.feature_dim
        assert xyz.shape[0] == feats.shape[0]
        self.observations.append((camera_pose, xyz, feats))
        world_xyz = trimesh.transform_points(xyz, camera_pose)

        if self.xyz is None:
            # Update current sparse voxel map.
            self.xyz = world_xyz
            self.feats = feats
        else:
            # Combine point clouds by adding in the current view to the previous ones and
            # voxelizing.
            self.xyz, self.feats = combine_point_clouds(
                self.xyz,
                self.feats,
                world_xyz,
                feats,
                sparse_voxel_size=self.resolution,
            )

    def get_data(self, in_place: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current point cloud and features; optionally copying."""
        if in_place or self.xyz is None:
            return self.xyz, self.feats
        else:
            return self.xyz.copy(), self.feats.copy()

    def recompute_map(self):
        """Recompute the entire map from scratch instead of doing incremental updates"""
        self.xyz, self.feats = None, None
        for camera_pose, xyz, feats in self.observations:
            world_xyz = trimesh.transform_points(xyz, camera_pose)
            if self.xyz is None:
                self.xyz, self.feats = world_xyz, feats
            else:
                pc_rgb = np.concatenate([self.feats, feats], axis=0)
                pc_xyz = np.concatenate([self.xyz, world_xyz], axis=0)
        pcd = numpy_to_pcd(pc_xyz, pc_rgb).voxel_down_sample(voxel_size=self.resolution)
        self.xyz, self.feats = pcd_to_numpy(pcd)
