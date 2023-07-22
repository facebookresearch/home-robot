from typing import Optional, Tuple

import numpy as np


def overlay_masks(
    masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Overlays the masks of objects
    Determines the order of masks based on mask size
    """
    mask_sizes = [np.sum(mask) for mask in masks]
    sorted_mask_idcs = np.argsort(mask_sizes)

    semantic_mask = np.zeros(shape)
    instance_mask = -np.ones(shape)
    for i_mask in sorted_mask_idcs[::-1]:  # largest to smallest
        semantic_mask[masks[i_mask].astype(bool)] = class_idcs[i_mask]
        instance_mask[masks[i_mask].astype(bool)] = i_mask

    return semantic_mask, instance_mask


def filter_depth(
    mask: np.ndarray, depth: np.ndarray, depth_threshold: Optional[float] = None
) -> np.ndarray:
    md = np.median(depth[mask == 1])  # median depth
    if md == 0:
        # Remove mask if more than half of points has invalid depth
        filter_mask = np.ones_like(mask, dtype=bool)
    elif depth_threshold is not None:
        # Restrict objects to 1m depth
        filter_mask = (depth >= md + depth_threshold) | (depth <= md - depth_threshold)
    else:
        filter_mask = np.zeros_like(mask, dtype=bool)
    mask_out = mask.copy()
    mask_out[filter_mask] = 0.0

    return mask_out
