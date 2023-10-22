import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
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
    pose: np.ndarray = None
    instance_id: Optional[int] = None
    object_coverage: Optional[int] = None

    def __init__(
        self,
        bbox,
        timestep,
        cropped_image,
        embedding,
        mask,
        point_cloud,
        pose,
        object_coverage,
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
        self.pose = pose
        self.category_id = category_id
        self.object_coverage = object_coverage


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
    local_id_to_global_id_map: List[Dict[int, int]] = []
    timesteps: List[int] = []

    def __init__(
        self,
        num_envs: int,
        du_scale: int,
        debug_visualize: bool = False,
        config=None,
        save_dir="instances",
        mask_cropped_instances=False,
        padding_cropped_instances=0,
        category_id_to_category_name=None,
    ):
        self.num_envs = num_envs
        self.du_scale = du_scale
        self.debug_visualize = debug_visualize
        self.mask_cropped_instances = mask_cropped_instances
        self.padding_cropped_instances = padding_cropped_instances
        self.category_id_to_category_name = category_id_to_category_name

        if config is not None:
            self.save_dir = os.path.join(
                config.DUMP_LOCATION, "instances", config.EXP_NAME
            )
        else:
            self.save_dir = save_dir

        if self.debug_visualize:
            shutil.rmtree(self.save_dir, ignore_errors=True)

        self.reset()

    def reset(self):
        self.images = [None for _ in range(self.num_envs)]
        self.point_cloud = [None for _ in range(self.num_envs)]
        self.instance_views = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.local_id_to_global_id_map = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def update_instance_id(
        self, env_id: int, local_instance_id: int, global_instance_id: int
    ):
        # fetch instance view from the list of unprocessed views
        # if global_instance_id already exists, add a new instance view to it
        # otherwise, create a new global instance with the given global_instance_id

        # get instance view
        instance_view = self.unprocessed_views[env_id].get(local_instance_id, None)
        if instance_view is None and self.debug_visualize:
            print(
                "instance view with local instance id",
                local_instance_id,
                "not found in unprocessed views",
            )

        # get global instance
        global_instance = self.instance_views[env_id].get(global_instance_id, None)
        if global_instance is None:
            # create a new global instance
            global_instance = Instance()
            global_instance.category_id = instance_view.category_id
            global_instance.instance_views.append(instance_view)
            self.instance_views[env_id][global_instance_id] = global_instance
        else:
            # add instance view to global instance
            global_instance.instance_views.append(instance_view)
        self.local_id_to_global_id_map[env_id][local_instance_id] = global_instance_id
        if self.debug_visualize:
            category_name = (
                f"cat_{instance_view.category_id}"
                if self.category_id_to_category_name is None
                else self.category_id_to_category_name[instance_view.category_id]
            )
            instance_write_path = os.path.join(
                self.save_dir, f"{global_instance_id}_{category_name}"
            )
            os.makedirs(instance_write_path, exist_ok=True)

            step = instance_view.timestep
            full_image = self.images[env_id][step]
            full_image = full_image.numpy().astype(np.uint8).transpose(1, 2, 0)
            full_image = full_image[..., ::-1]
            # overlay mask on image
            mask = np.zeros(full_image.shape, full_image.dtype)
            mask[:, :] = (0, 0, 255)
            mask = cv2.bitwise_and(mask, mask, mask=instance_view.mask.astype(np.uint8))
            masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
            cv2.imwrite(
                os.path.join(
                    instance_write_path,
                    f"step_{self.timesteps[env_id]}_local_id_{local_instance_id}.png",
                ),
                masked_image,
            )
            # print(
            #     "mapping local instance id",
            #     local_instance_id,
            #     "to global instance id",
            #     global_instance_id,
            # )

    def process_instances_for_env(
        self,
        env_id: int,
        semantic_map: torch.Tensor,
        instance_map: torch.Tensor,
        point_cloud: torch.Tensor,
        pose: torch.Tensor,
        image: torch.Tensor,
        num_sem_categories: int,
    ):
        # create a dict for mapping instance ids to categories
        instance_id_to_category_id = {}

        self.unprocessed_views[env_id] = {}
        self.local_id_to_global_id_map[env_id] = {}
        # append image to list of images
        if self.images[env_id] is None:
            self.images[env_id] = image.unsqueeze(0).detach().cpu()
        else:
            self.images[env_id] = torch.cat(
                [self.images[env_id], image.unsqueeze(0).detach().cpu()], dim=0
            )
        if self.point_cloud[env_id] is None:
            self.point_cloud[env_id] = point_cloud.unsqueeze(0).detach().cpu()
        else:
            self.point_cloud[env_id] = torch.cat(
                [self.point_cloud[env_id], point_cloud.unsqueeze(0).detach().cpu()],
                dim=0,
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
            category_id = category_id[0].item()
            # print(instance_id, category_id)

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

            if self.mask_cropped_instances:
                masked_image = image * instance_mask
            else:
                masked_image = image

            # get cropped image
            p = self.padding_cropped_instances
            h, w = masked_image.shape[1:]
            cropped_image = (
                masked_image[
                    :,
                    max(bbox[0, 0] - p, 0) : min(bbox[1, 0] + p, h),
                    max(bbox[0, 1] - p, 0) : min(bbox[1, 1] + p, w),
                ]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            instance_mask = instance_mask.cpu().numpy().astype(bool)

            # get embedding
            embedding = None

            # get point cloud
            point_cloud_instance = point_cloud[instance_mask_downsampled.cpu().numpy()]

            object_coverage = np.sum(instance_mask) / instance_mask.size

            # get instance view
            instance_view = InstanceView(
                bbox=bbox,
                timestep=self.timesteps[env_id],
                cropped_image=cropped_image,
                embedding=embedding,
                mask=instance_mask,
                point_cloud=point_cloud_instance.cpu().numpy(),
                category_id=category_id,
                pose=pose.detach().cpu(),
                object_coverage=object_coverage
            )

            # append instance view to list of instance views
            self.unprocessed_views[env_id][instance_id.item()] = instance_view
            # save cropped image with timestep in filename
            if self.debug_visualize:
                os.makedirs(f"{self.save_dir}/all", exist_ok=True)
                cv2.imwrite(
                    f"{self.save_dir}/all/{self.timesteps[env_id] + 1}_{instance_id.item()}.png",
                    cropped_image[:,:,::-1],
                )

        self.timesteps[env_id] += 1

    def get_unprocessed_instances_per_env(self, env_id: int):
        return self.unprocessed_views[env_id]

    def process_instances(
        self,
        semantic_channels: torch.Tensor,
        instance_channels: torch.Tensor,
        point_cloud: torch.Tensor,
        pose: torch.Tensor,
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
                pose[env_id],
                image[env_id],
                semantic_map.shape[1],
            )

    def reset_for_env(self, env_id: int):
        self.instance_views[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0
        self.local_id_to_global_id_map[env_id] = {}
