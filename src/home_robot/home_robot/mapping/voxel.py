# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

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
