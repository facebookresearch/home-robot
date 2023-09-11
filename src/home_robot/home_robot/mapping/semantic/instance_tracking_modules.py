# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.ops import box3d_overlap
from torch import Tensor

from home_robot.core.interfaces import Observations
from home_robot.utils.bboxes_3d import (
    box3d_intersection_from_bounds,
    box3d_nms,
    box3d_overlap_from_bounds,
    box3d_volume_from_bounds,
    get_box_bounds_from_verts,
    get_box_verts_from_bounds,
)
from home_robot.utils.image import dilate_or_erode_mask
from home_robot.utils.point_cloud_torch import get_bounds


@dataclass
class InstanceView:
    """
    Stores information about a single view of a single instance
    """

    # Required: 2D and 3D bbox. Should we unify the names?
    bbox: Tensor
    """ [4,] bbox: bounding box of instance in the current image """
    bounds: Tensor
    """[3, 2] xyz mins and maxes"""
    """TODO: rename to bounds_3d"""
    timestep: int
    """ timestep: timestep at which the current view was recorded """

    # View info
    cropped_image: Optional[Tensor] = None
    """ cropped_image: cropped image of instance in the current image"""
    embedding: Optional[Tensor] = None
    """ embedding: embedding of instance in the current image """
    mask: Tensor = None
    """ mask: mask of instance in the current (uncropped) image """
    image_instance_id: Optional[int] = None
    """ID of this instance in the image"""

    # Detection info
    global_instance_id: Optional[int] = None
    """ID of this instance in the scene (across views)"""
    category_id: Optional[int] = None
    """category_id: category id of instance in the current image"""
    score: Optional[float] = None
    """score: confidence of the detection (used for NMS)"""

    # 3D info
    point_cloud: Tensor = None
    point_cloud_rgb: Tensor = None
    point_cloud_features: Tensor = None
    """point_cloud: 3d locations of world points corresponding to instance view pixels"""
    """TODO: rename to points_3d"""


@dataclass
class Instance:
    """
    A single instance found in the environment. Each instance is composed of a list of InstanceView objects, each of which is a view of the instance at a particular timestep.
    """

    name: str = None
    category_id: int = None
    """ingeger indicating the category"""
    point_cloud: Tensor = None
    """point_cloud: aggregated point cloud for the instance """
    point_cloud_rgb: Tensor = None
    """point_cloud_rgb: aggregated point cloud colors for the instance """
    point_cloud_features: Tensor = None
    """point_cloud_features: aggregated point cloud features for the instance """
    bounds: Tensor = None
    """ 3 x 2 mins and maxes """
    instance_views: List[InstanceView] = field(default_factory=list)
    """List of all instance views"""
    score: float = None
    """Confidence score of bbox detection"""
    score_aggregation_method: str = "max"

    def add_instance_view(self, instance_view: InstanceView):
        if len(self.instance_views) == 0:
            # instantiate from instance
            self.category_id = instance_view.category_id
            self.instance_views.append(instance_view)
            self.bounds = instance_view.bounds
            self.point_cloud = instance_view.point_cloud
            self.point_cloud_rgb = instance_view.point_cloud_rgb
            self.point_cloud_features = instance_view.point_cloud_features
            self.score = instance_view.score
        else:
            # Right now we concatenate point clouds
            # To keep the number of points manageable, we could make the pointcloud a VoxelizedPointcloud class
            self.point_cloud = torch.cat(
                [self.point_cloud, instance_view.point_cloud], dim=0
            )
            if self.point_cloud_rgb is not None:
                self.point_cloud_rgb = torch.cat(
                    [self.point_cloud_rgb, instance_view.point_cloud_rgb],
                    dim=0,
                )
            if self.point_cloud_features is not None:
                self.point_cloud_features = torch.cat(
                    [
                        self.point_cloud_features,
                        instance_view.point_cloud_features,
                    ],
                    dim=0,
                )
            if self.score_aggregation_method == "max":
                self.score = max(self.score, instance_view.score)
            elif self.score_aggregation_method == "mean":
                self.score = (
                    self.score * len(self.instance_views) + instance_view.score
                ) / (len(self.instance_views) + 1)
            else:
                raise NotImplementedError(
                    f'Unknown score_aggregation_method "{self.score_aggregation_method}"'
                )

            # add instance view to global instance
            # do this last because we use the current length for computing average score above
            self.instance_views.append(instance_view)

            self.bounds = get_bounds(self.point_cloud)

    def _show_point_cloud_pytorch3d(self, **plot_scene_kwargs):
        """Visualize an instance in the map

        Args:
            idx (int): Instance index

        Returns:
            ptc_fig: Plotly visualization of pointcloud
        """
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
        from home_robot.utils.data_tools.dict import update

        # Show points
        features = [self.point_cloud_rgb] if self.point_cloud_rgb is not None else None
        ptc = Pointclouds(points=[self.point_cloud], features=features)

        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=200_000,
        )
        fig = plot_scene(
            plots={
                f"Name {self.name}: (category: {self.category_id}) -- {len(self.instance_views)} views": {
                    "Points": ptc,
                    # "Instance boxes": detected_boxes,
                    # "Fused boxes": global_boxes,
                    # "cameras": cameras,
                },
                # Could add keyframes or instances here.
            },
            **update(_default_plot_args, plot_scene_kwargs),
        )
        return fig

    def _show_instance_view_frames(self, mask_out_opacity=0.3):
        import matplotlib.pyplot as plt

        plt.figure()

        n_views = len(self.instance_views)
        f, axarr = plt.subplots(n_views, 1)  # n_views rows, 1 col
        for i, view in enumerate(self.instance_views):
            ax = axarr[i] if (n_views > 1) else axarr
            cropped_image_np = torch.from_numpy(
                view.cropped_image.detach().cpu().numpy()
            )

            mask_np = torch.from_numpy(view.mask.detach().cpu().numpy())
            if (
                mask_np.shape[0] != cropped_image_np.shape[0]
                or mask_np.shape[1] != cropped_image_np.shape[1]
            ):
                mask_np = mask_np[
                    view.bbox[0, 0] : view.bbox[1, 0], view.bbox[0, 1] : view.bbox[1, 1]
                ]
            display_im = cropped_image_np * mask_out_opacity + (
                1 - mask_out_opacity
            ) * (cropped_image_np * mask_np[..., None])

            ax.imshow(display_im)
            ax.title.set_text(f"View {i}")
        plt.tight_layout()
        plt.show()
        return f


class InstanceMemory:
    """
    InstanceMemory stores information about instances found in the environment. It stores a list of Instance objects, each of which is a single instance found in the environment.

    images: list of egocentric images at each timestep
    instance_views: list of InstanceView objects at each timestep
    point_cloud: list of point clouds at each timestep
    unprocessed_views: list of unprocessed InstanceView objects at each timestep, before they are added to an Instance object
    timesteps: list of timesteps
    """

    images: List[Tensor] = []
    instances: List[Dict[int, Instance]] = []
    point_cloud: List[Tensor] = []
    unprocessed_views: List[Dict[int, InstanceView]] = []
    timesteps: List[int] = []

    def __init__(
        self,
        num_envs: int,
        du_scale: int,
        instance_association: str = "bbox_iou",
        instance_association_within_class: bool = True,
        iou_threshold: float = 0.8,
        global_box_nms_thresh: float = 0.0,
        debug_visualize: bool = False,
        erode_mask_num_pix: int = 0,
        erode_mask_num_iter: int = 1,
        instance_view_score_aggregation_mode="max",
        overlap_eps: float = 1e-6,
        min_pixels_for_instance_view=100,
    ):
        """See class definition for information about InstanceMemory

        Args:
            num_envs (int): Number of environments to track
            du_scale (int): Downsample images by 1 / du_scale
            instance_association (str, optional): Instance association method. Defaults to "bbox_iou".
            iou_threshold (float, optional): Threshold for associating instance views to global memory. Defaults to 0.8.
            global_box_nms_thresh (float): If nonzero, perform nonmax suppression on global bboxes after associating instances to memory
            debug_visualize (bool, optional): Visualize by writing out to disk. Defaults to False.
            erode_mask_num_pix (int, optional): If nonzero, how many pixels to erode instance masks. Defaults to 0.
            erode_mask_num_iter (int, optional): If erode_mask_num_pix is nonzero, how times to iterate erode instance masks. Defaults to 1.
            instance_view_score_aggregation_mode (str): When adding views to an instance, how to update instance scores. Defaults to 'max'
        """
        self.num_envs = num_envs
        self.du_scale = du_scale
        self.debug_visualize = debug_visualize
        self.instance_association = instance_association
        self.iou_threshold = iou_threshold
        self.erode_mask_num_pix = erode_mask_num_pix
        self.erode_mask_num_iter = erode_mask_num_iter
        self.global_box_nms_thresh = global_box_nms_thresh
        self.instance_view_score_aggregation_mode = instance_view_score_aggregation_mode
        self.overlap_eps = overlap_eps
        self.min_pixels_for_instance_view = min_pixels_for_instance_view
        self.instance_association_within_class = instance_association_within_class

        if self.debug_visualize:
            import shutil

            shutil.rmtree("instances/", ignore_errors=True)
        self.reset()

    def reset(self):
        """
        Reset the state of instance memory after an episode ends
        """
        self.images = [None for _ in range(self.num_envs)]
        self.point_cloud = [None for _ in range(self.num_envs)]
        self.instances = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def get_instance(self, env_id: int, global_instance_id: int) -> Instance:
        """
        Retrieve an instance given an environment ID and a global instance ID.

        Args:
            env_id (int): The environment ID.
            global_instance_id (int): The global instance ID within the specified environment.

        Returns:
            Instance: The Instance object associated with the given IDs.
        """
        return self.instances[env_id][global_instance_id]

    def get_global_instance_ids(self, env_id: int) -> List[int]:
        """
        Get the list of global instance IDs associated with a given environment ID.

        Args:
            env_id (int): The environment ID.

        Returns:
            List[int]: A list of global instance IDs associated with the environment.
        """
        return list(self.instances[env_id].keys())

    def get_instances_by_ids(
        self, env_id: int, global_instance_idxs: Optional[Sequence[int]] = None
    ) -> List[Instance]:
        """
        Retrieve a list of instances for a given environment ID. If instance indexes are specified, only instances
        with those indexes will be returned.

        Args:
            env_id (int): The environment ID.
            global_instance_idxs (Optional[Sequence[int]]): The global instance IDs to retrieve. If None, all instances
                                                    for the given environment will be returned. Defaults to None.

        Returns:
            List[Instance]: List of Instance objects associated with the given IDs.
        """
        if global_instance_idxs is None:
            return list(self.instances[env_id].values())
        return [self.instances[env_id][g_id] for g_id in global_instance_idxs]

    def pop_global_instance(
        self, env_id: int, global_instance_id: int, skip_reindex: bool = False
    ) -> Instance:
        """
        Remove and return an instance given an environment ID and a global instance ID. Optionally skip reindexing of
        global instance IDs.

        Args:
            env_id (int): The environment ID.
            global_instance_id (int): The global instance ID to remove.
            skip_reindex (bool): Whether to skip reindexing of global instance IDs. Defaults to False.

        Returns:
            Instance: The removed Instance object.
        """
        instance = self.instances[env_id].pop(global_instance_id)
        if not skip_reindex:
            self.reindex_global_instances()
        return instance

    def reindex_global_instances(self, env_id: int) -> Dict[int, Instance]:
        """
        Reindex the global instance IDs within a specified environment. This is typically used after removing an instance
        to ensure that the global instance IDs are contiguous. Mutates self.instances.

        Args:
            env_id (int): The environment ID for which to reindex global instance IDs.

        Returns:
            Dict[int, Instance]: The newly indexed dictionary of Instance objects for the given environment.
        """
        ids, instances = zip(*self.instances[env_id].items())
        new_ids = range(len(ids))
        new_env_instances = dict(zip(new_ids, instances))
        self.instances[env_id] = new_env_instances
        return self.instances[env_id]

    def get_ids_to_instances(
        self, env_id: int, category_id: Optional[int] = None
    ) -> List[Instance]:
        """
        Retrieve a Dict of IDs -> global instances for a given environment. If category_id is specified,
        only instances matching that category will be returned.

        Args:
            env_id (int): The environment ID.
            category_id: (Optional[Sequence[int]]): The category of instances to retreive. If None, all instances
                                                    for the given environment will be returned. Defaults to None.

        Returns:
            Dict[int, Instance]: ID -> Instance objects associated with the given category_id.
        """
        # Get global instances
        global_instance_ids = self.get_global_instance_ids(env_id)
        if len(global_instance_ids) == 0:
            return []
        global_instances = self.get_instances_by_ids(
            env_id=env_id, global_instance_idxs=global_instance_ids
        )
        return_dict = {
            gid: inst for gid, inst in zip(global_instance_ids, global_instances)
        }
        if category_id is not None:
            return_dict = {
                gid: g for gid, g in return_dict.items() if g.category_id == category_id
            }
        return return_dict

    def find_global_instance_by_bbox_overlap(
        self,
        env_id: int,
        local_instance_id: int,
        match_within_category: bool = False,
    ) -> Optional[int]:
        """
        Find the global instance with the maximum bounding box IoU overlap above a certain threshold with a local instance in a specific environment.

        This method helps identify the global instance that shares the highest spatial overlap with a local instance across multiple views,
        based on their 3D bounding boxes and Intersection over Union (IoU) scores.

        Args:
            env_id (int): Identifier for the environment in which the search is conducted.
            local_instance_id (int): Identifier for the local instance within the specified environment.
            iou_threshold (float): Minimum IoU threshold for considering instances as matching candidates.
            match_within_category (bool): Only associate w/ bboxes of same category_id. Defaults to False.

        Returns:
            matching_global_instance (Optional[int]): Global instance ID with the maximum bounding box IoU overlap above the threshold,
                or None if no instances meet the criteria.

        Note:
            The method calculates IoU scores between the bounding box of the local instance and the bounding boxes of the global instances.
            It then selects the instance with the highest IoU score above the specified threshold as the matching global instance.
            If no instances meet the criteria, the method returns None.
        """

        if len(self.instances[env_id]) == 0:
            return None
        # global_instance_ids, global_bounds = zip(
        #     *[
        #         (inst_id, instance.bounds)
        #         for inst_id, instance in self.instances[env_id].items()
        #     ]
        # )
        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)
        if instance_view is not None:
            ids_to_instances = self.get_ids_to_instances(
                env_id,
            )
            if len(ids_to_instances) == 0:
                return None
            global_bounds = torch.stack(
                [inst.bounds for inst in ids_to_instances.values()], dim=0
            )

            _, iou = box3d_overlap_from_bounds(
                instance_view.bounds.unsqueeze(0),
                global_bounds,
                self.overlap_eps,
            )
            ious = iou.flatten()
            if ious.max() > self.iou_threshold:
                return list(ids_to_instances.keys())[ious.argmax()]
        return None

    def find_global_instance_by_one_sided_iou(
        self, env_id: int, local_instance_id: int, match_within_category: bool = False
    ) -> Optional[int]:
        """
        Find the global instance ID that has the maximum one-sided Intersection over Union (IoU)
        with the local instance identified by the local_instance_id in the environment specified by env_id.

        Args:
            env_id (int): The environment ID.
            local_instance_id (int): The local instance ID whose global counterpart needs to be found.
            match_within_category (bool): Only associate w/ bboxes of same category_id. Defaults to False.

        Returns:
            Optional[int]: The global instance ID with the maximum one-sided IoU.
                        Returns None if no such global instance is found or if the maximum IoU
                        is below the threshold.
        """
        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)
        match_category_id = instance_view.category_id if match_within_category else None
        volume1 = box3d_volume_from_bounds(instance_view.bounds)
        assert torch.all(volume1 > 0.0), instance_view.bounds

        if instance_view is not None:
            ids_to_instances = self.get_ids_to_instances(
                env_id, category_id=match_category_id
            )
            if len(ids_to_instances) == 0:
                return None
            global_bounds = torch.stack(
                [inst.bounds for inst in ids_to_instances.values()], dim=0
            )

            vol_int, _ = box3d_overlap_from_bounds(
                instance_view.bounds.unsqueeze(0), global_bounds, self.overlap_eps
            )

            ious = vol_int / volume1
            assert ious.ndim == 2 and ious.shape[0] == 1, ious.shape
            ious = ious.flatten()

            if ious.max() > self.iou_threshold:
                return list(ids_to_instances.keys())[ious.argmax()]
        return None

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
        raise NotImplementedError
        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_view_id)
        volume1 = box3d_volume_from_bounds(instance_view.bounds)

        if instance_view is not None:
            global_instance_ids = self.get_global_instance_ids(env_id)
            if len(global_instance_ids) == 0:
                return None
            global_bounds = [
                inst.bounds
                for inst in self.get_instances_by_ids(
                    env_id=env_id, global_instance_idxs=global_instance_ids
                )
            ]

            global_bounds = torch.stack(global_bounds, dim=0)
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

    def associate_instances_to_memory(self):
        """
        Associate instance views with existing instances or create new instances based on matching criteria.

        This method performs instance association for each instance view across environments. It determines whether an instance view
        should be added to an existing instance or a new instance should be created.

        The association process can be based on Intersection over Union (IoU) or a global map.

        For each environment and local instance view, the following steps are taken:
        - If the instance association method is set to "bbox_iou", the best matching global instance is found using the
        `find_global_instance_by_bbox_overlap` method. If a suitable global instance is not found (IoU below a threshold),
        a new instance is created. Otherwise, the instance view is associated with the existing global instance.
        - If the instance association method is set to "map_overlap", the association occurs during the global map update, and no action is taken here.
        - If the instance association method is not recognized, a NotImplementedError is raised.

        Note:
            The instance association process is critical for maintaining a coherent memory representation of instances across views.

        Raises:
            NotImplementedError: When an unrecognized instance association method is specified.
        """
        if self.instance_association == "bbox_iou":
            for env_id in range(self.num_envs):
                for local_instance_id, instance_view in self.unprocessed_views[
                    env_id
                ].items():
                    if self.instance_association == "bbox_iou":
                        global_instance_id = self.find_global_instance_by_bbox_overlap(
                            env_id,
                            local_instance_id,
                            match_within_category=self.instance_association_within_class,
                        )
                        if global_instance_id is None:
                            global_instance_id = len(self.instances[env_id])
                        self.add_view_to_instance(
                            env_id, local_instance_id, global_instance_id
                        )
        if self.instance_association == "bbox_one_sided_iou":
            for env_id in range(self.num_envs):
                for local_instance_id, instance_view in self.unprocessed_views[
                    env_id
                ].items():
                    if self.instance_association == "bbox_one_sided_iou":
                        global_instance_id = self.find_global_instance_by_one_sided_iou(
                            env_id,
                            local_instance_id,
                            match_within_category=self.instance_association_within_class,
                        )
                        if global_instance_id is None:
                            global_instance_id = len(self.instances[env_id])
                        self.add_view_to_instance(
                            env_id, local_instance_id, global_instance_id
                        )
        elif self.instance_association == "map_overlap":
            # association happens at the time of global map update
            pass
        else:
            raise NotImplementedError

        if self.global_box_nms_thresh > 0.0:
            for env_id in range(self.num_envs):
                self.global_instance_nms(env_id)

    def global_instance_nms(
        self,
        env_id: int,
        nms_iou_thresh: Optional[float] = None,
        within_category: bool = True,
    ) -> None:
        """
        Perform Non-Maximum Suppression (NMS) on global instances within the specified environment.

        This function performs NMS based on 3D bounding boxes and confidence scores. Instances that have a high overlap (IoU)
        and lower confidence scores are removed, and their instance views are added to the instance that is kept. The instances
        are reindexed after the NMS operation.

        Args:
            env_id: The environment ID where the global instances are located.
            nms_iou_thresh (Optional[float]): The IoU threshold for performing NMS.
                If not provided, the function uses the global_box_nms_thresh
                class variable as the threshold.
            within_category (bool): Only do NMS on objects of the same category

        Returns:
            None: This function modifies the internal state of the InstanceMemory object but does not return any value.

        Example usage:
            >>> instance_memory.global_instance_nms(env_id=0, nms_iou_thresh=0.5)

        Note:
            This function directly modifies the internal list of instances (`self.instances`) and
            reindexes them after the NMS operation.
        """
        if nms_iou_thresh is None:
            nms_iou_thresh = self.global_box_nms_thresh
        assert 0.0 < nms_iou_thresh and nms_iou_thresh <= 1.0, nms_iou_thresh
        if within_category:
            categories = torch.tensor(
                [inst.category_id for inst in self.get_instances_by_ids(env_id)]
            ).unique()
            for category_id in categories:
                category_id = int(category_id)
                ids_to_instances = self.get_ids_to_instances(
                    env_id, category_id=category_id
                )
                ids, instances = list(ids_to_instances.keys()), list(
                    ids_to_instances.values()
                )
                instance_bounds = torch.stack(
                    [inst.bounds for inst in instances], dim=0
                )
                confidences = torch.stack([inst.score for inst in instances], dim=0)
                # Could implement a different NMS designed for partial views
                # Idea: Go over bounding boxes large-to-small
                # If intersection / small_volume > thresh -- assign small -> large
                confidences = box3d_volume_from_bounds(
                    instance_bounds
                )  # Larger boxes get priority
                keep, vol, iou, assignments = box3d_sub_box_suppression(
                    instance_bounds,
                    confidence_score=confidences,
                    iou_threshold=nms_iou_thresh,
                )
                for keep_id, delete_ids in assignments.items():
                    # Map the sequential ids from box3d_nms to the instanceMemory global id
                    keep_id = ids[keep_id]
                    delete_ids = [ids[d_id] for d_id in delete_ids]
                    inst_to_keep = self.get_instance(
                        env_id=env_id, global_instance_id=int(keep_id)
                    )
                    for delete_id in delete_ids:
                        inst_to_delete = self.pop_global_instance(
                            env_id, global_instance_id=int(delete_id), skip_reindex=True
                        )
                        for view in inst_to_delete.instance_views:
                            inst_to_keep.add_instance_view(view)
        else:
            global_instance_ids = self.get_global_instance_ids(env_id)
            instances = self.get_instances_by_ids(env_id, global_instance_ids)
            instance_bounds = torch.stack([inst.bounds for inst in instances], dim=0)
            instance_corners = get_box_verts_from_bounds(instance_bounds)
            confidences = torch.stack([inst.score for inst in instances], dim=0)
            keep, vol, iou, assignments = box3d_nms(
                instance_corners,
                confidence_score=confidences,
                iou_threshold=nms_iou_thresh,
            )
            for keep_id, delete_ids in assignments.items():
                inst_to_keep = self.get_instance(
                    env_id=env_id, global_instance_id=int(keep_id)
                )
                for delete_id in delete_ids:
                    inst_to_delete = self.pop_global_instance(
                        env_id, global_instance_id=int(delete_id), skip_reindex=True
                    )
                    for view in inst_to_delete.instance_views:
                        inst_to_keep.add_instance_view(view)
        self.reindex_global_instances(env_id)

    def global_instance_nms_ids(
        self,
        env_id: int,
        global_ids: Sequence[int],
        nms_iou_thresh: Optional[float] = None,
    ):
        pass

    def get_local_instance_view(self, env_id: int, local_instance_id: int):
        """
        Retrieve the local instance view associated with a specific local instance in a given environment.

        This method fetches the unprocessed instance view corresponding to a specified local instance within a particular environment.

        Args:
            env_id (int): Identifier for the environment in which the instance view should be retrieved.
            local_instance_id (int): Identifier for the local instance within the specified environment.

        Returns:
            instance_view (Optional[InstanceView]): The instance view associated with the specified local instance in the given environment,
                or None if no matching instance view is found.
        """
        instance_view = self.unprocessed_views[env_id].get(local_instance_id, None)
        if instance_view is None and self.debug_visualize:
            print(
                "instance view with local instance id",
                local_instance_id,
                "not found in unprocessed views",
            )
        return instance_view

    def add_view_to_instance(
        self, env_id: int, local_instance_id: int, global_instance_id: int
    ):
        """
        Update instance associations and memory based on instance view information.

        This method handles the process of updating instance associations and memory based on instance view information.
        It ensures that the appropriate global instances are maintained or created and that their attributes are updated.

        Args:
            env_id (int): Identifier for the environment in which the update is performed.
            local_instance_id (int): Identifier for the local instance view within the specified environment.
            global_instance_id (int): Identifier for the global instance to which the local instance view will be associated.

        Note:
            - If the global instance with the given `global_instance_id` does not exist, a new global instance is created.
            - If the global instance already exists, the instance view is added to it, and its attributes are updated accordingly.

        Debugging:
            If the `debug_visualize` flag is enabled, the method generates visualizations for the instance association process
            and saves them to disk in the "instances" directory. Debug information is printed to the console.
        """

        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)

        # get global instance
        global_instance = self.instances[env_id].get(global_instance_id, None)
        if global_instance is None:
            global_instance = Instance(
                score_aggregation_method=self.instance_view_score_aggregation_mode
            )
            self.instances[env_id][global_instance_id] = global_instance
        global_instance.add_instance_view(instance_view)

        if self.debug_visualize:
            import os

            import cv2

            step = instance_view.timestep
            full_image = self.images[env_id][step]
            full_image = (
                (full_image * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            # overlay mask on image
            mask = np.zeros(full_image.shape, full_image.dtype)
            mask[:, :] = (0, 0, 255)
            mask = cv2.bitwise_and(mask, mask, mask=instance_view.mask.astype(np.uint8))
            masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
            os.makedirs(f"instances/{global_instance_id}", exist_ok=True)
            cv2.imwrite(
                f"instances/{global_instance_id}/{self.timesteps[env_id]}_{local_instance_id}_cat_{instance_view.category_id}.png",
                masked_image,
            )
            print(
                "mapping local instance id",
                local_instance_id,
                "to global instance id",
                global_instance_id,
            )

    def process_instances_for_env(
        self,
        env_id: int,
        instance_seg: Tensor,
        point_cloud: Tensor,
        image: Tensor,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        mask_out_object: bool = True,
        background_instance_label: int = 0,
        valid_points: Optional[Tensor] = None,
    ):
        """
        Process instance information in the current frame and add instance views to the list of unprocessed views for future association.

        This method processes instance information from instance segmentation, point cloud data, and images for a given environment.
        It extracts and prepares instance views based on the provided data and adds them to the list of unprocessed views for later association.

        Args:
            env_id (int): Identifier for the environment being processed.
            instance_seg (Tensor): [H, W] tensor of instance ids at each pixel
            point_cloud (Tensor): Point cloud data in world coordinates.
            image (Tensor): [3, H, W] RGB image
            instance_classes (Optional[Tensor]): [K,] class ids for each instance in instance seg
                class_int = instance_classes[instance_id]
            instance_scores (Optional[Tensor]): [K,] detection confidences for each instance in instance_seg
            mask_out_object (bool): true if we want to save crops of just objects on black background; false otherwise
                # If false does it not save crops? Not black background?
            background_class_label(int): id indicating background points in instance_seg. That view is not saved. (default = 0)
            valid_points (Tensor): [H, W] boolean tensor indicating valid points in the pointcloud
        Note:
            - The method creates instance views for detected instances within the provided data.
            - If a semantic segmentation tensor is provided, each instance is associated with a semantic category.
            - Instance views are added to the `unprocessed_views` dictionary for later association with specific instances.

        Debugging:
            If the `debug_visualize` flag is enabled, cropped images and visualization data are saved to disk.
        """
        # create a dict for mapping instance ids to categories
        instance_id_to_category_id = {}
        assert (
            image.shape[0] == 3
        ), "Ensure that RGB images are channels-first and in the right format."

        self.unprocessed_views[env_id] = {}
        # append image to list of images
        if self.images[env_id] is None:
            self.images[env_id] = image.unsqueeze(0)
        else:
            self.images[env_id] = torch.cat(
                [self.images[env_id], image.unsqueeze(0)], dim=0
            )
        if self.point_cloud[env_id] is None:
            self.point_cloud[env_id] = point_cloud.unsqueeze(0)
        else:
            self.point_cloud[env_id] = torch.cat(
                [self.point_cloud[env_id], point_cloud.unsqueeze(0)], dim=0
            )

        # Valid opints
        if valid_points is None:
            valid_points = torch.full(
                image.shape[:, 0], True, dtype=torch.bool, device=image.device
            )
        if self.du_scale != 1:
            valid_points_downsampled = (
                torch.nn.functional.interpolate(
                    valid_points.unsqueeze(0).unsqueeze(0).float(),
                    scale_factor=1 / self.du_scale,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .bool()
            )
        else:
            valid_points_downsampled = valid_points

        # unique instances
        instance_ids = torch.unique(instance_seg)
        for instance_id in instance_ids:
            # skip background
            if instance_id == background_instance_label:
                continue
            # get instance mask
            instance_mask = instance_seg == instance_id

            # get semantic category
            category_id = None
            if instance_classes is not None:
                category_id = instance_classes[instance_id]

            # detection score
            score = None
            if instance_scores is not None:
                score = instance_scores[instance_id]

            instance_id_to_category_id[instance_id] = category_id

            # get bounding box
            bbox = torch.stack(
                [
                    instance_mask.nonzero().min(dim=0)[0],
                    instance_mask.nonzero().max(dim=0)[0] + 1,
                ]
            )

            assert bbox.shape == (
                2,
                2,
            ), "Bounding box has extra dimensions - you have a problem with input instance image mask!"

            # TODO: If we use du_scale, we should apply this at the beginning to speed things up
            if self.du_scale != 1:
                # downsample mask by du_scale using "NEAREST"
                instance_mask_downsampled = (
                    torch.nn.functional.interpolate(
                        instance_mask.unsqueeze(0).unsqueeze(0).float(),
                        scale_factor=1 / self.du_scale,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )

            else:
                instance_mask_downsampled = instance_mask

            # Erode instance masks for point cloud
            # TODO: We can do erosion and masking on the downsampled/cropped image to avoid unnecessary computation
            if self.erode_mask_num_pix > 0:
                instance_mask_downsampled = dilate_or_erode_mask(
                    instance_mask_downsampled.unsqueeze(0),
                    radius=-self.erode_mask_num_pix,
                    num_iterations=self.erode_mask_num_iter,
                ).squeeze(0)
                instance_mask = dilate_or_erode_mask(
                    instance_mask.unsqueeze(0),
                    radius=-self.erode_mask_num_pix,
                    num_iterations=self.erode_mask_num_iter,
                ).squeeze(0)

            # Mask out the RGB image using the original detection mask
            if mask_out_object:
                masked_image = image * instance_mask
            else:
                masked_image = image
            image_box = masked_image[
                :, bbox[0, 0] : bbox[1, 0], bbox[0, 1] : bbox[1, 1]
            ]
            # get cropped image
            cropped_image = image_box.permute(1, 2, 0)

            # get embedding
            embedding = None

            # get point cloud
            point_mask_downsampled = (
                instance_mask_downsampled & valid_points_downsampled
            )
            point_cloud_instance = point_cloud[point_mask_downsampled]
            point_cloud_rgb_instance = image.permute(1, 2, 0)[point_mask_downsampled]

            n_points = point_mask_downsampled.sum()
            n_mask = instance_mask_downsampled.sum()

            # Create InstanceView if the view is large enough
            if n_mask >= self.min_pixels_for_instance_view and n_points > 1:
                bounds = get_bounds(point_cloud_instance)
                volume = float(box3d_volume_from_bounds(bounds).squeeze())

                if volume < 1e-6:
                    warnings.warn(
                        f"Skipping box with {n_points} points in cloud and {n_points} points in mask and {volume} volume",
                        UserWarning,
                    )
                else:
                    # get instance view
                    instance_view = InstanceView(
                        bbox=bbox,
                        timestep=self.timesteps[env_id],
                        cropped_image=cropped_image,  # .cpu().numpy(),
                        embedding=embedding,
                        mask=instance_mask,  # cpu().numpy().astype(bool),
                        point_cloud=point_cloud_instance,  # .cpu().numpy(),
                        point_cloud_rgb=point_cloud_rgb_instance,
                        category_id=category_id,
                        score=score,
                        bounds=bounds,  # .cpu().numpy(),
                    )
                    # append instance view to list of instance views
                    self.unprocessed_views[env_id][instance_id.item()] = instance_view

            # save cropped image with timestep in filename
            if self.debug_visualize:
                import os

                import cv2

                os.makedirs("instances/", exist_ok=True)
                cv2.imwrite(
                    f"instances/{self.timesteps[env_id]}_{instance_id.item()}.png",
                    (cropped_image.cpu().numpy() * 255).astype(np.uint8),
                )

        self.timesteps[env_id] += 1

    def get_unprocessed_instances_per_env(self, env_id: int):
        return self.unprocessed_views[env_id]

    def process_instances(
        self,
        instance_channels: Tensor,
        point_cloud: Tensor,
        image: Tensor,
        semantic_channels: Optional[Tensor] = None,
    ):
        """
        Process instance information across environments and associate instance views with global instances.

        This method processes instance information from instance channels, point cloud data, and images across different environments.
        It extracts and prepares instance views based on the provided data for each environment and associates them with global instances.

        Args:
            instance_channels (Tensor): Tensor containing instance segmentation channels for each environment.
            point_cloud (Tensor): Tensor containing point cloud data for each environment.
            image (Tensor): Tensor containing image data for each environment.
            semantic_channels (Optional[Tensor]): Tensor containing semantic segmentation channels for each environment, if available.

        Note:
            - Instance views are extracted and prepared for each environment based on the instance channels.
            - If semantic segmentation channels are provided, each instance view is associated with a semantic category.
            - Instance views are added to the list of unprocessed views for later association with specific instances.
            - After processing instance views for all environments, the method associates them with global instances using `associate_instances_to_memory()`.

        Debugging:
            If the `debug_visualize` flag is enabled, cropped images and visualization data are saved to disk.
        """
        instance_segs = instance_channels.argmax(dim=1).int()
        semantic_segs = None
        if semantic_channels is not None:
            semantic_segs = semantic_channels.argmax(dim=1).int()
        for env_id in range(self.num_envs):
            semantic_seg = None if semantic_segs is None else semantic_segs[env_id]
            self.process_instances_for_env(
                env_id,
                instance_segs[env_id],
                point_cloud[env_id],
                image[env_id],
                semantic_seg=semantic_seg,
            )
        self.associate_instances_to_memory()

    def reset_for_env(self, env_id: int):
        self.instances[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0


def box3d_sub_box_suppression(
    bounds: Tensor, confidence_score: Tensor, iou_threshold: float = 0.3
):
    """
    Non-max suppression where boxes inside other boxes get suppressed

    Args:
      bounds: (N, 3, 2) vertex coordinates. Must be in order specified by box3d_overlap
      confidence_score: (N,)
      iou_threshold: Suppress boxes whose IoU > iou_threshold

    Returns:
      keep, vol, iou, assignments
      keep: indexes into N of which bounding boxes to keep
      vol: (N, N) tensor of the volume of the intersecting convex shapes
      iov: (N, M) tensor of the intersection over union which is
          defined as: `iou = vol_int / (vol)`
      assignments: superbox_idx -> List[boxes_to_delete]
    """
    assert len(bounds) > 0, bounds.shape

    order = torch.argsort(confidence_score)

    (
        intersection,
        _,
        _,
    ) = box3d_intersection_from_bounds(bounds, bounds)
    volume = box3d_volume_from_bounds(bounds)
    intersection_over_volume = intersection / volume.unsqueeze(0)
    # There is a subtle possible bug here if we want to combine bboxes instead of just suppress
    # I.e. if box 1 contains most of box 8, but doensn't contain box1
    # and box 8 contains box 1
    # and box 1 is larger than box 8 larger than box 1
    # Then box 8 gets assigned to 1 and box 1 makes it through
    # Is this desired behavior?

    keep = []
    assignments = {}
    while len(order) > 0:

        idx = order[-1]  # Highest confidence (S)

        # push S in filtered predictions list
        keep.append(idx)

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # keep the boxes with IoU less than thresh_iou
        # _, iou = box3d_overlap(bounding_boxes[idx].unsqueeze(0), bounding_boxes[order])
        mask = intersection_over_volume[idx, order] < iou_threshold
        assignments[idx] = order[~mask]
        order = order[mask]

    return torch.tensor(keep), intersection, intersection_over_volume, assignments
