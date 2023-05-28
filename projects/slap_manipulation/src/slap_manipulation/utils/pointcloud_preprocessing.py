"""
This is only for processing inference-time data at the moment.
TODO:   Make these methods generic enough to support H5 data processing as well.
        Right now the data processing in dataloaders occures using class's native methods
"""
from typing import List, Tuple

import numpy as np

from home_robot.utils.point_cloud import show_point_cloud


def crop_around_voxel(
    feat: np.ndarray, xyz: np.ndarray, crop_location: np.ndarray, crop_size: float
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
    breakpoint()
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
