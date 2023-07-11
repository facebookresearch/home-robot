from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class InstanceView:
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
        self.bbox = bbox
        self.timestep = timestep
        self.cropped_image = cropped_image
        self.embedding = embedding
        self.mask = mask
        self.point_cloud = point_cloud
        self.category_id = category_id


class Instance:
    def __init__(self):
        self.name = None
        self.category_id = None
        self.instance_views = []


class InstanceMemory:
    images: List[torch.Tensor] = []
    instance_views: List[Dict[int, Instance]] = []
    point_cloud: List[torch.Tensor] = []
    unprocessed_views: List[Dict[int, InstanceView]] = []
    timesteps: List[int] = []

    def __init__(self, num_envs: int, du_scale: int):
        self.num_envs = num_envs
        self.du_scale = du_scale
        self.reset()

    def reset(self):
        self.images = [None for _ in range(self.num_envs)]
        self.point_cloud = [None for _ in range(self.num_envs)]
        self.instance_views = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def update_instance_id(
        self, env_id: int, local_instance_id: int, global_instance_id: int
    ):
        # fetch instance view from the list of unprocessed views
        # if global_instance_id already exists, add a new instance view to it
        # otherwise, create a new global instance with the given global_instance_id

        # get instance view
        instance_view = self.unprocessed_views[env_id][local_instance_id]
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
            bbox = torch.stack(
                [
                    instance_mask.nonzero().min(dim=0)[0],
                    instance_mask.nonzero().max(dim=0)[0],
                ]
            ).cpu().numpy()
            # get cropped image
            cropped_image = image[:, bbox[0, 0] : bbox[1, 0], bbox[0, 1] : bbox[1, 1]].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # get embedding
            embedding = None

            # downsample mask by du_scale using "NEAREST"
            instance_mask = (
                torch.nn.functional.interpolate(
                    instance_mask.unsqueeze(0).unsqueeze(0).float(),
                    scale_factor=1 / self.du_scale,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .bool()
            ).cpu().numpy()
            # get point cloud
            point_cloud_instance = point_cloud[instance_mask]

            # get instance view
            instance_view = InstanceView(
                bbox=bbox,
                timestep=self.timesteps[env_id],
                cropped_image=cropped_image,
                embedding=embedding,
                mask=instance_mask,
                point_cloud=point_cloud_instance,
                category_id=category_id,
            )

            # append instance view to list of instance views
            self.unprocessed_views[env_id][instance_id.item()] = instance_view

        self.timesteps[env_id] += 1
        return instance_id_to_category_id

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

    def reset_for_env(self, env_id: int):
        self.instance_views[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0
