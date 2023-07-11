# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This is only for processing inference-time data at the moment.
TODO:   Make these methods generic enough to support H5 data processing as well.
        Right now the data processing in dataloaders occures using class's native methods
"""
from typing import List, Optional, Tuple

import numpy as np
from slap_manipulation.utils.data_visualizers import (
    show_point_cloud_with_keypt_and_closest_pt,
    show_semantic_mask,
)

from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud


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


def voxelize_and_get_interaction_point(
    xyz,
    rgb,
    feats,
    interaction_ee_keyframe,
    voxel_size=0.01,
    debug=False,
    semantic_id=1,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """uniformly voxelizes the input point-cloud and returns the closest-point
    in the point-cloud to the task's interaction ee-keyframe"""
    # voxelize another time to get sampled version
    input_pcd = numpy_to_pcd(xyz, rgb)
    (
        voxelized_pcd,
        _,
        voxelized_index_trace_vectors,
    ) = input_pcd.voxel_down_sample_and_trace(
        voxel_size, input_pcd.get_min_bound(), input_pcd.get_max_bound()
    )
    voxelized_index_trace = []
    for intvec in voxelized_index_trace_vectors:
        voxelized_index_trace.append(np.asarray(intvec))
    voxelized_xyz = np.asarray(voxelized_pcd.points)
    voxelized_rgb = np.asarray(voxelized_pcd.colors)
    voxelized_feats = aggregate_feats(feats, voxelized_index_trace)

    if debug:
        show_semantic_mask(voxelized_xyz, voxelized_rgb, voxelized_feats, semantic_id)

    # for the voxelized pcd
    if voxelized_xyz.shape[0] < 10:
        return (None, None, None, None, None, None)
    voxelized_pcd_tree = o3d.geometry.KDTreeFlann(voxelized_pcd)
    # Find closest points based on ref_ee_keyframe
    # This is used to supervise the location when we're detecting where the action
    # could have happened
    [_, target_idx_1, _] = voxelized_pcd_tree.search_knn_vector_3d(
        interaction_ee_keyframe[:3, 3], 1
    )
    target_idx_down_pcd = np.asarray(target_idx_1)[0]
    closest_pt_down_pcd = voxelized_xyz[target_idx_down_pcd]

    # this is for exact point
    # @Priyam I do not think the following is really needed
    input_pcd_tree = o3d.geometry.KDTreeFlann(input_pcd)
    [_, target_idx_2, _] = input_pcd_tree.search_knn_vector_3d(
        # ee_keyframe[:3, 3], 1
        interaction_ee_keyframe[:3, 3],
        1,
    )
    target_idx_og_pcd = np.asarray(target_idx_2)[0]
    closest_pt_og_pcd = xyz[target_idx_og_pcd]

    if debug:
        print("Closest point in voxelized pcd")
        show_point_cloud_with_keypt_and_closest_pt(
            voxelized_xyz,
            voxelized_rgb,
            interaction_ee_keyframe[:3, 3],
            interaction_ee_keyframe[:3, :3],
            voxelized_xyz[target_idx_down_pcd].reshape(3, 1),
        )
        print("Closest point in original pcd")
        show_point_cloud_with_keypt_and_closest_pt(
            xyz,
            rgb,
            interaction_ee_keyframe[:3, 3],
            interaction_ee_keyframe[:3, :3],
            xyz[target_idx_og_pcd].reshape(3, 1),
        )
    return (
        voxelized_xyz,
        voxelized_rgb,
        voxelized_feats,
        target_idx_down_pcd,
        closest_pt_down_pcd,
        target_idx_og_pcd,
        closest_pt_og_pcd,
    )


def aggregate_feats(feats: np.ndarray, downsampled_index_trace: List[np.ndarray]):
    """
    this method is used to aggregate features over a downsampled point-cloud
    while using its index-trace (from Open3D)
    feats: (N, feat_dim)
    downsampled_index_trace: list of list of index
    """
    # downsampled_index_trace is a list of list of index
    # average feats over each list of index
    agg_feats = []
    _, feat_dim = feats.shape
    for idx in downsampled_index_trace:
        most_freq_feat = np.bincount(feats[idx, :].reshape(-1)).argmax()
        agg_feats.append(most_freq_feat)
    agg_feats = np.stack(agg_feats, axis=0).reshape(-1, feat_dim)
    return agg_feats
