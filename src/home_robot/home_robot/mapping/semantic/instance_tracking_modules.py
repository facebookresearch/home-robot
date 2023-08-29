# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch3d.ops import box3d_overlap
from torch import Tensor

from home_robot.core.interfaces import Observations
from home_robot.utils.bboxes_3d import (
    box3d_overlap_from_bounds,
    get_box_bounds_from_verts,
    get_box_verts_from_bounds,
)
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


class Instance:
    """
    A single instance found in the environment. Each instance is composed of a list of InstanceView objects, each of which is a view of the instance at a particular timestep.
    """

    def __init__(self):
        """
        Initialize Instance

        name: name of instance
        category_id: category id of instance
        point_cloud: aggregated point cloud for the instance
        bounds Tuple[np.ndarray, np.ndarray]: bounding box of the aggregated point cloud
        instance_views: list of InstanceView objects
        """
        self.name = None
        self.category_id = None
        self.point_cloud = None
        self.point_cloud_rgb = None
        self.point_cloud_features = None
        self.bounds = None
        self.instance_views = []

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

    images: List[torch.Tensor] = []
    instance_views: List[Dict[int, Instance]] = []
    point_cloud: List[torch.Tensor] = []
    unprocessed_views: List[Dict[int, InstanceView]] = []
    timesteps: List[int] = []

    def __init__(
        self,
        num_envs: int,
        du_scale: int,
        instance_association: str = "bbox_iou",
        iou_threshold: float = 0.8,
        debug_visualize: bool = False,
    ):
        self.num_envs = num_envs
        self.du_scale = du_scale
        self.debug_visualize = debug_visualize
        self.instance_association = instance_association
        self.iou_threshold = iou_threshold
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
        self.instance_views = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def get_bbox_overlap_np(
        self,
        local_bbox: Tuple[np.ndarray, np.ndarray],
        global_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    ):
        """
        Calculate Intersection over Union (IoU) scores between a local 3D bounding box and a list of global 3D bounding boxes.

        Args:
            local_bbox (Tuple[np.ndarray, np.ndarray]): Bounding box of a point cloud obtained from the local instance in the current frame.
            global_bboxes (List[Tuple[np.ndarray, np.ndarray]]): List of bounding boxes of instances obtained by aggregating point clouds across different views.

        Returns:
            ious (np.ndarray): IoU scores between the local_bbox and each of the global_bboxes.
        """
        global_bboxes_min, global_bboxes_max = zip(*global_bboxes)
        global_bboxes_min = np.stack(global_bboxes_min, axis=0)
        global_bboxes_max = np.stack(global_bboxes_max, axis=0)
        intersection_min = np.maximum(
            np.expand_dims(local_bbox[0], 0), global_bboxes_min
        )
        intersection_max = np.minimum(
            np.expand_dims(local_bbox[1], 0), global_bboxes_max
        )
        zero_iou = (intersection_min > intersection_max).any(axis=-1)
        intersection = np.prod(intersection_max - intersection_min, axis=-1)
        union = (
            np.prod(global_bboxes_max - global_bboxes_min, axis=-1)
            + np.prod(local_bbox[1] - local_bbox[0])
            - intersection
        )
        ious = intersection / union
        ious[zero_iou] = 0.0
        ious[np.isnan(ious)] = 0.0
        return ious

    def find_global_instance_by_bbox_overlap(
        self, env_id: int, local_instance_id: int
    ) -> Optional[int]:
        """
        Find the global instance with the maximum bounding box IoU overlap above a certain threshold with a local instance in a specific environment.

        This method helps identify the global instance that shares the highest spatial overlap with a local instance across multiple views,
        based on their 3D bounding boxes and Intersection over Union (IoU) scores.

        Args:
            env_id (int): Identifier for the environment in which the search is conducted.
            local_instance_id (int): Identifier for the local instance within the specified environment.
            iou_threshold (float): Minimum IoU threshold for considering instances as matching candidates.

        Returns:
            matching_global_instance (Optional[int]): Global instance ID with the maximum bounding box IoU overlap above the threshold,
                or None if no instances meet the criteria.

        Note:
            The method calculates IoU scores between the bounding box of the local instance and the bounding boxes of the global instances.
            It then selects the instance with the highest IoU score above the specified threshold as the matching global instance.
            If no instances meet the criteria, the method returns None.
        """

        if len(self.instance_views[env_id]) == 0:
            return None
        global_instance_ids, global_bounds = zip(
            *[
                (inst_id, instance.bounds)
                for inst_id, instance in self.instance_views[env_id].items()
            ]
        )
        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)
        if instance_view is not None:
            _, iou = box3d_overlap_from_bounds(
                instance_view.bounds.unsqueeze(0), torch.stack(global_bounds, dim=0)
            )
            ious = iou.flatten()
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
                            env_id, local_instance_id
                        )
                        if global_instance_id is None:
                            global_instance_id = len(self.instance_views[env_id])
                        self.add_view_to_instance(
                            env_id, local_instance_id, global_instance_id
                        )

        elif self.instance_association == "map_overlap":
            # association happens at the time of global map update
            pass
        else:
            raise NotImplementedError

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
        global_instance = self.instance_views[env_id].get(global_instance_id, None)
        if global_instance is None:
            # create a new global instance
            global_instance = Instance()
            global_instance.category_id = instance_view.category_id
            global_instance.instance_views.append(instance_view)
            global_instance.bounds = instance_view.bounds
            global_instance.point_cloud = instance_view.point_cloud
            global_instance.point_cloud_rgb = instance_view.point_cloud_rgb
            global_instance.point_cloud_features = instance_view.point_cloud_features
            self.instance_views[env_id][global_instance_id] = global_instance
        else:
            # add instance view to global instance
            global_instance.instance_views.append(instance_view)
            # Right now we concatenate point clouds
            # To keep the number of points manageable, we could do voxel downsampling
            global_instance.point_cloud = torch.cat(
                [global_instance.point_cloud, instance_view.point_cloud], dim=0
            )
            if global_instance.point_cloud_rgb is not None:
                global_instance.point_cloud_rgb = torch.cat(
                    [global_instance.point_cloud_rgb, instance_view.point_cloud_rgb],
                    dim=0,
                )
            if global_instance.point_cloud_features is not None:
                global_instance.point_cloud_features = torch.cat(
                    [
                        global_instance.point_cloud_features,
                        instance_view.point_cloud_features,
                    ],
                    dim=0,
                )
            global_instance.bounds = get_bounds(global_instance.point_cloud)

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
        instance_seg: torch.Tensor,
        point_cloud: torch.Tensor,
        image: torch.Tensor,
        instance_classes: Optional[torch.Tensor] = None,
        instance_scores: Optional[torch.Tensor] = None,
        mask_out_object: bool = True,
        background_instance_label: int = 0,
    ):
        """
        Process instance information in the current frame and add instance views to the list of unprocessed views for future association.

        This method processes instance information from instance segmentation, point cloud data, and images for a given environment.
        It extracts and prepares instance views based on the provided data and adds them to the list of unprocessed views for later association.

        Args:
            env_id (int): Identifier for the environment being processed.
            instance_seg (torch.Tensor): [H, W] tensor of instance ids at each pixel
            point_cloud (torch.Tensor): Point cloud data in world coordinates.
            image (torch.Tensor): [3, H, W] RGB image
            instance_classes (Optional[torch.Tensor]): [K,] class ids for each instance in instance seg
                class_int = instance_classes[instance_id]
            instance_scores (Optional[torch.Tensor]): [K,] detection confidences for each instance in instance_seg
            mask_out_object (bool): true if we want to save crops of just objects on black background; false otherwise
                # If false does it not save crops? Not black background?
            background_class_label(int): id indicating background points in instance_seg. That view is not saved. (default = 0)

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
            point_cloud_instance = point_cloud[instance_mask_downsampled]
            point_cloud_rgb_instance = image.permute(1, 2, 0)[instance_mask_downsampled]

            if instance_mask_downsampled.sum() > 0:
                bounds = get_bounds(point_cloud_instance)
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
        instance_channels: torch.Tensor,
        point_cloud: torch.Tensor,
        image: torch.Tensor,
        semantic_channels: Optional[torch.Tensor] = None,
    ):
        """
        Process instance information across environments and associate instance views with global instances.

        This method processes instance information from instance channels, point cloud data, and images across different environments.
        It extracts and prepares instance views based on the provided data for each environment and associates them with global instances.

        Args:
            instance_channels (torch.Tensor): Tensor containing instance segmentation channels for each environment.
            point_cloud (torch.Tensor): Tensor containing point cloud data for each environment.
            image (torch.Tensor): Tensor containing image data for each environment.
            semantic_channels (Optional[torch.Tensor]): Tensor containing semantic segmentation channels for each environment, if available.

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
        self.instance_views[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0
