from typing import List, Tuple

import numpy as np


def crop_around_voxel(feat, xyz, crop_location, crop_size):
    """Crop a point cloud around given voxel"""
    mask = np.linalg.norm(xyz - crop_location, axis=1) < crop_size
    return xyz[mask, :], feat[mask, :]


def get_local_action_prediction_problem(
    cfg,
    feat: np.ndarray,
    xyz: np.ndarray,
    p_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # crop from og pcd and mean-center it
    crop_xyz, crop_rgb = crop_around_voxel(xyz, feat, p_i, cfg.local_problem_size)
    crop_xyz = crop_xyz - p_i[None].repeat(crop_xyz.shape[0], axis=0)
    # show_point_cloud(crop_xyz, crop_rgb, orig=np.zeros(3))
    if crop_rgb.shape[0] > cfg.num_pts:
        # Downsample pt clouds
        downsample = np.arange(crop_rgb.shape[0])
        np.random.shuffle(downsample)
        if cfg.num_pts != -1:
            downsample = downsample[: cfg.num_pts]
        crop_rgb = crop_rgb[downsample]
        crop_xyz = crop_xyz[downsample]
    status = True
    if crop_xyz.shape[0] < 10:
        status = False
    return crop_xyz, crop_rgb, status
