from enum import Enum, auto
from typing import List, Optional, Union

import torch
from torch import Tensor

from home_robot.utils.bboxes_3d import (
    box3d_intersection_from_bounds,
    box3d_overlap_from_bounds,
    box3d_volume_from_bounds,
)

from .core import Instance, InstanceView


class Bbox3dOverlapMethodEnum(Enum):
    IOU = "IOU"
    ONE_SIDED_IOU = "ONE_SIDED_IOU"


def get_bbox_similarity(
    bounds1: Union[Tensor, List[Tensor]],
    bounds2: Union[Tensor, List[Tensor]],
    overlap_eps: float = 1e-6,
    mode: Bbox3dOverlapMethodEnum = Bbox3dOverlapMethodEnum.ONE_SIDED_IOU,
) -> Tensor:
    if len(bounds1) == 0:
        return None
    if len(bounds2) == 0:
        return None
    if not isinstance(bounds1, Tensor):
        bounds1 = torch.stack(bounds1, dim=0)
    if not isinstance(bounds2, Tensor):
        bounds2 = torch.stack(bounds2, dim=0)

    if mode == Bbox3dOverlapMethodEnum.ONE_SIDED_IOU:
        volume1 = box3d_volume_from_bounds(bounds1)
        assert torch.all(volume1 > 0.0), bounds1
        vol_int, _ = box3d_overlap_from_bounds(bounds1, bounds2, overlap_eps)
        ious = vol_int / volume1.unsqueeze(1)
    elif mode == Bbox3dOverlapMethodEnum.IOU:
        _, ious = box3d_overlap_from_bounds(bounds1, bounds2, overlap_eps)
    else:
        raise NotImplementedError(f"Unsupported Bbox3dOverlapMethodEnum mode: {mode}")
    assert ious.ndim == 2 and ious.shape[0] == len(bounds1), ious.shape
    return ious


class EncoderSimilarityMethodEnum(Enum):
    MAX = auto()
    # MEAN = auto()


def dot_product_similarity(feats1, feats2, normalize=True):
    """
    Calculate the cosine similarity between two sets of feature vectors.

    Args:
        feats1: NxD tensor (N: number of vectors, D: dimensionality of each vector)
        feats2: MxD tensor (M: number of vectors, D: dimensionality of each vector)
        normalize: Whether to normalize the input feature vectors. Default is True.

    Returns:
        N x M tensor of similarities
    """
    if feats1 is None or len(feats1) == 0:
        return None
    if feats2 is None or len(feats2) == 0:
        return None
    if not isinstance(feats1, Tensor):
        feats1 = torch.stack(feats1, dim=0)
    if not isinstance(feats2, Tensor):
        feats2 = torch.stack(feats2, dim=0)
    if normalize:
        # Normalize the input feature vectors to have unit L2 norm
        feats1 = feats1 / torch.norm(feats1, dim=1, keepdim=True)
        feats2 = feats2 / torch.norm(feats2, dim=1, keepdim=True)

    # Calculate the dot product between the (optionally) normalized feature vectors
    dot_product = torch.mm(feats1, feats2.t())

    return dot_product


# Geometry-based matching
def find_global_instance_by_pointcloud_overlap(
    self, env_id: int, local_instance_view_id: int
) -> Optional[int]:
    """
    Find the global instance ID that has the most overlapping points in the point cloud
    with the local instance identified by `local_instance_id` in the environment specified by `env_id`.

    This function performs the following steps to associate the local instance with a global instance:
    1. Compute the 3D box intersection between the local instance's 3D bounding box and those of all global instances.
    2. Filter both local and global instance point clouds by this intersection.
    3. Compute the distance to the nearest global points from the local instance's point cloud.
            nearest_global_point = knn(instance_view.points_filtered, global_instance.points_filtered)
    4. Determine the percentage of points in the local instance's filtered point cloud that are near to points in the global instances' point clouds.
            points_matched = % of instance_view.points_filtered[nearest_point_dist < dist_thresh]
    5. Associate the local instance with the global instance based on one of the following metrics:
        - The (% matched points) * one-sided IoU: one_sided_IoU * points_matched.mean()
        - The sum of matched points * points_matched.sum()

    Args:
        env_id (int): The environment ID.
        local_instance_view_id (int): The local instance view ID whose global counterpart needs to be found.

    Returns:
        Optional[int]: The global instance ID with the most point cloud overlap.
                    Returns None if no such global instance is found.

    TODO:
        - Optimize by having global instances store a voxelized point cloud to keep the number of points manageable.
    """
    raise NotImplementedError(
        "Placeholder pending correct implementation of geometry based matching"
    )
    # get instance view
    instance_view = self.get_local_instance_view(env_id, local_instance_view_id)
    volume1 = box3d_volume_from_bounds(instance_view.bounds)

    if instance_view is not None:
        global_instance_ids = self.get_global_instance_ids(env_id)
        if len(global_instance_ids) == 0:
            return None
        instances = self.get_instances_by_ids(env_id, global_instance_ids)
        global_bounds = torch.stack([inst.bounds for inst in instances], dim=0)
        vol_int, iou, intersection_bounds = box3d_intersection_from_bounds(
            instance_view.bounds.unsqueeze(0), global_bounds, self.overlap_eps
        )
        # 2. Filter by intersection_bounds
        # 3. nearest_global_point = knn(instance_view.points_filtered, global_instance.points_filtered)
        # 4. points_matched = % of instance_view.points_filtered[nearest_point_dist < dist_thresh]
        # 5.
        ious = vol_int / volume1
        assert ious.ndim == 2 and ious.shape[0] == 1, ious.shape
        ious = ious.flatten()

        if ious.max() > self.iou_threshold:
            return global_instance_ids[ious.argmax()]
    return None
