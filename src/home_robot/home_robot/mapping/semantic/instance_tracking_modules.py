# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class InstanceView:
    """
    Stores information about a single view of an instance

    bbox: bounding box of instance in the current image
    timestep: timestep at which the current view was recorded
    cropped_image: cropped image of instance in the current image
    embedding: embedding of instance in the current image
    mask: mask of instance in the current image
    point_cloud: point cloud of instance in the current image
    category_id: category id of instance in the current image
    """

    bbox: Tuple[int, int, int, int]
    timestep: int
    cropped_image: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    # mask of instance in the current image
    mask: np.ndarray = None
    # point cloud of instance in the current image
    point_cloud: np.ndarray = None
    category_id: Optional[int] = None

    def __init__(
        self,
        bbox,
        timestep,
        cropped_image,
        embedding,
        mask,
        point_cloud,
        category_id=None,
    ):
        """
        Initialize InstanceView
        """
        self.bbox = bbox
        self.timestep = timestep
        self.cropped_image = cropped_image
        self.embedding = embedding
        self.mask = mask
        self.point_cloud = point_cloud
        self.category_id = category_id


class Instance:
    """
    A single instance found in the environment. Each instance is composed of a list of InstanceView objects, each of which is a view of the instance at a particular timestep.
    """

    def __init__(self):
        """
        Initialize Instance

        name: name of instance
        category_id: category id of instance
        instance_views: list of InstanceView objects
        """
        self.name = None
        self.category_id = None
        self.point_cloud = None
        self.bounds = None
        self.instance_views = []


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
        instance_association: str = "iou",
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
        self.images = [None for _ in range(self.num_envs)]
        self.point_cloud = [None for _ in range(self.num_envs)]
        self.instance_views = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def get_bbox_overlap(
        self,
        local_bbox: Tuple[np.ndarray, np.ndarray],
        global_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    ):
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

    def find_global_instance_by_bbox_overlap(self, env_id: int, local_instance_id: int):
        if len(self.instance_views[env_id]) == 0:
            return None, None
        global_instance_ids, global_bounds = zip(
            *[
                (inst_id, instance.bounds)
                for inst_id, instance in self.instance_views[env_id].items()
            ]
        )
        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)
        if instance_view is not None:
            ious = self.get_bbox_overlap(instance_view.bounds, global_bounds)
            if ious.max() > self.iou_threshold:
                return global_instance_ids[ious.argmax()], ious.max()
        return None, None

    def associate_instances_to_memory(self):
        # for each instance view, find the best matching instance
        # if the best matching instance is above a threshold, add the instance view to the instance
        # otherwise, create a new instance
        for env_id in range(self.num_envs):
            for local_instance_id, instance_view in self.unprocessed_views[
                env_id
            ].items():
                if self.instance_association == "iou":
                    global_instance_id, iou = self.find_global_instance_by_bbox_overlap(
                        env_id, local_instance_id
                    )
                    if global_instance_id is None:
                        global_instance_id = len(self.instance_views[env_id])
                    self.update_instance_id(
                        env_id, local_instance_id, global_instance_id, iou
                    )
                elif self.instance_association == "map":
                    # association happens at the time of global map update
                    pass
                else:
                    raise NotImplementedError

    def get_local_instance_view(self, env_id: int, local_instance_id: int):
        instance_view = self.unprocessed_views[env_id].get(local_instance_id, None)
        if instance_view is None and self.debug_visualize:
            print(
                "instance view with local instance id",
                local_instance_id,
                "not found in unprocessed views",
            )
        return instance_view

    def update_instance_id(
        self, env_id: int, local_instance_id: int, global_instance_id: int, iou
    ):
        # fetch instance view from the list of unprocessed views
        # if global_instance_id already exists, add a new instance view to it
        # otherwise, create a new global instance with the given global_instance_id

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
            self.instance_views[env_id][global_instance_id] = global_instance
        else:
            # add instance view to global instance
            global_instance.instance_views.append(instance_view)
            global_instance.point_cloud = np.concatenate(
                [global_instance.point_cloud, instance_view.point_cloud], axis=0
            )
            global_instance.bounds = np.min(
                global_instance.point_cloud, axis=0
            ), np.max(global_instance.point_cloud, axis=0)

        if self.debug_visualize:
            import os

            import cv2

            step = instance_view.timestep
            full_image = self.images[env_id][step]
            full_image = full_image.numpy().astype(np.uint8).transpose(1, 2, 0)
            # overlay mask on image
            mask = np.zeros(full_image.shape, full_image.dtype)
            mask[:, :] = (0, 0, 255)
            mask = cv2.bitwise_and(mask, mask, mask=instance_view.mask.astype(np.uint8))
            masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
            os.makedirs(f"instances/{global_instance_id}", exist_ok=True)
            cv2.imwrite(
                f"instances/{global_instance_id}/{self.timesteps[env_id]}_{local_instance_id}_cat_{instance_view.category_id}_{iou}.png",
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
        semantic_map: torch.Tensor,
        instance_map: torch.Tensor,
        point_cloud: torch.Tensor,
        image: torch.Tensor,
        num_sem_categories: int,
    ):
        # create a dict for mapping instance ids to categories
        instance_id_to_category_id = {}

        self.unprocessed_views[env_id] = {}
        # append image to list of images
        if self.images[env_id] is None:
            self.images[env_id] = image.unsqueeze(0).cpu()
        else:
            self.images[env_id] = torch.cat(
                [self.images[env_id], image.unsqueeze(0).cpu()], dim=0
            )
        if self.point_cloud[env_id] is None:
            self.point_cloud[env_id] = point_cloud.unsqueeze(0)
        else:
            self.point_cloud[env_id] = torch.cat(
                [self.point_cloud[env_id], point_cloud.unsqueeze(0)], dim=0
            )
        # unique instances
        instance_ids = torch.unique(instance_map)
        for instance_id in instance_ids:
            # skip background
            if instance_id == 0:
                continue
            # get instance mask
            instance_mask = instance_map == instance_id

            # get semantic category
            category_id = semantic_map[instance_mask].unique()
            category_id = category_id[0]

            instance_id_to_category_id[instance_id] = category_id

            # get bounding box
            bbox = (
                torch.stack(
                    [
                        instance_mask.nonzero().min(dim=0)[0],
                        instance_mask.nonzero().max(dim=0)[0] + 1,
                    ]
                )
                .cpu()
                .numpy()
            )

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

            masked_image = image * instance_mask
            # get cropped image
            cropped_image = (
                masked_image[:, bbox[0, 0] : bbox[1, 0], bbox[0, 1] : bbox[1, 1]]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            instance_mask = instance_mask.cpu().numpy().astype(bool)

            # get embedding
            embedding = None

            # get point cloud
            point_cloud_instance = (
                point_cloud[instance_mask_downsampled.cpu().numpy()].cpu().numpy()
            )

            if instance_mask_downsampled.sum() > 0 and point_cloud_instance.sum() > 0:
                bounds = np.min(point_cloud_instance, axis=0), np.max(
                    point_cloud_instance, axis=0
                )

                # get instance view
                instance_view = InstanceView(
                    bbox=bbox,
                    timestep=self.timesteps[env_id],
                    cropped_image=cropped_image,
                    embedding=embedding,
                    mask=instance_mask,
                    point_cloud=point_cloud_instance,
                    category_id=category_id,
                    bounds=bounds,
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
                    cropped_image,
                )

        self.timesteps[env_id] += 1

    def get_unprocessed_instances_per_env(self, env_id: int):
        return self.unprocessed_views[env_id]

    def process_instances(
        self,
        semantic_channels: torch.Tensor,
        instance_channels: torch.Tensor,
        point_cloud: torch.Tensor,
        image: torch.Tensor,
    ):
        instance_map = instance_channels.argmax(dim=1).int()
        semantic_map = semantic_channels.argmax(dim=1).int()
        for env_id in range(self.num_envs):
            self.process_instances_for_env(
                env_id,
                semantic_map[env_id],
                instance_map[env_id],
                point_cloud[env_id],
                image[env_id],
                semantic_map.shape[1],
            )
        self.associate_instances_to_memory()

    def reset_for_env(self, env_id: int):
        self.instance_views[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0
