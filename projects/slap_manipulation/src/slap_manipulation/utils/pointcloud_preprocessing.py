# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This is only for processing inference-time data at the moment.
TODO:   Make these methods generic enough to support H5 data processing as well.
        Right now the data processing in dataloaders occures using class's native methods
"""
from typing import List, Tuple

import numpy as np

from home_robot.utils.point_cloud import show_point_cloud


def crop_around_voxel(
    feat: np.ndarray,
    xyz: np.ndarray,
    crop_location: np.ndarray,
    crop_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a point cloud around given voxel"""
    mask = np.linalg.norm(xyz - crop_location, axis=1) < crop_size
    return xyz[mask, :], feat[mask, :]


def get_local_action_prediction_problem(
    cfg,
    feat: np.ndarray,
    xyz: np.ndarray,
    p_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns a cropped version of the input point-cloud mean-centered around the predicted
    interaction point (p_i)
    """
    # crop from og pcd and mean-center it
    crop_xyz, crop_feat = crop_around_voxel(feat, xyz, p_i, cfg.query_radius)
    crop_xyz = crop_xyz - p_i[None].repeat(crop_xyz.shape[0], axis=0)
    # show_point_cloud(crop_xyz, crop_feat, orig=np.zeros(3))
    if crop_feat.shape[0] > cfg.num_pts:
        # Downsample pt clouds
        downsample = np.arange(crop_feat.shape[0])
        np.random.shuffle(downsample)
        if cfg.num_pts != -1:
            downsample = downsample[: cfg.num_pts]
        crop_feat = crop_feat[downsample]
        crop_xyz = crop_xyz[downsample]
    status = True
    if crop_xyz.shape[0] < 10:
        status = False
    return crop_feat, crop_xyz, status


def find_closest_point_to_line(xyz, gripper_position, action_axis):
    line_to_points = xyz - gripper_position
    projections_on_line = line_to_points * action_axis
    closest_points_on_line = gripper_position + projections_on_line * line_to_points
    distances_to_line = np.linalg.norm(closest_points_on_line - xyz, axis=-1)
    closest_indices = np.argmin(distances_to_line)
    return closest_indices, xyz[closest_indices]


def filter_and_remove_duplicate_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    feats: np.ndarray,
    depth: np.ndarray = None,
    voxel_size: float = 0.001,
    semantic_id: float = 0,
    debug_voxelization: bool = False,
):
    """filters out points based on depth and them removes duplicate/overlapping
    points by voxelizing at a fine resolution"""
    # heuristic based trimming
    if depth is not None:
        mask = np.bitwise_and(depth < 1.5, depth > 0.3)
        rgb = rgb[mask]
        xyz = xyz[mask]
        if feats is not None:
            feats = feats[mask]
        z_mask = xyz[:, 2] > 0.15
        rgb = rgb[z_mask]
        xyz = xyz[z_mask]
        if feats is not None:
            feats = feats[z_mask]
    if np.any(rgb > 1.0):
        rgb = rgb / 255.0
    debug_views = False
    if debug_views:
        print("xyz", xyz.shape)
        print("rgb", rgb.shape)
        show_point_cloud(xyz, rgb)

    # voxelize at a granular voxel-size rather than random downsample
    pcd = numpy_to_pcd(xyz, rgb)
    (
        pcd_voxelized,
        _,
        voxelized_index_trace_vectors,
    ) = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
    )
    voxelized_index_trace = []
    for intvec in voxelized_index_trace_vectors:
        voxelized_index_trace.append(np.asarray(intvec))
    rgb = np.asarray(pcd_voxelized.colors)
    xyz = np.asarray(pcd_voxelized.points)
    if feats is not None:
        feats = aggregate_feats(feats, voxelized_index_trace)

    if debug_voxelization:
        if feats is not None:
            show_semantic_mask(xyz, rgb, feats)

    return xyz, rgb, feats


def voxelize_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    feat: np.ndarray = None,
    debug_voxelization: bool = False,
    voxel_size: float = 0.01,
):
    """voxelizes point-cloud and aggregates features"""
    # voxelize at a granular voxel-size rather than random downsample
    pcd = numpy_to_pcd(xyz, rgb)
    (
        pcd_voxelized,
        _,
        voxelized_index_trace_vectors,
    ) = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
    )
    voxelized_index_trace = []
    for intvec in voxelized_index_trace_vectors:
        voxelized_index_trace.append(np.asarray(intvec))
    rgb = np.asarray(pcd_voxelized.colors)
    xyz = np.asarray(pcd_voxelized.points)
    if feat is not None:
        feat = aggregate_feats(feat, voxelized_index_trace)

    if debug_voxelization:
        show_semantic_mask(xyz, rgb, feats=feat)

    return xyz, rgb, feat
