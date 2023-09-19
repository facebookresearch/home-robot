# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    Creates a SparseVoxelMap of a ScanNet scene and evaluates it on that scene
"""
import logging
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluation.obj_det import eval_bboxes_and_print
from hydra_zen import store, zen
from torch import Tensor
from tqdm import tqdm

from home_robot.agent.ovmm_agent.ovmm_perception import OvmmPerception
from home_robot.core.interfaces import Observations
from home_robot.datasets.scannet import ScanNetDataset
from home_robot.mapping.semantic.instance_tracking_modules import Instance
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.perception.constants import RearrangeDETICCategories

logger = logging.getLogger(__name__)


class SemanticVocab(IntEnum):
    FULL = auto()
    SIMPLE = auto()
    ALL = auto()


class SparseVoxelMapAgent:
    """Simple class to collect RGB, Depth, and Pose information for building 3d spatial-semantic
    maps for the robot. Needs to subscribe to:
    - color images
    - depth images
    - camera info
    - joint states/head camera pose
    - base pose (relative to world frame)

    - Add option to cache Instance Segmentation + Pixel features
    This is an example collecting the data; not necessarily the way you should do it.
    """

    def __init__(
        self,
        semantic_sensor: Optional[OvmmPerception] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        visualize_planner=False,
        device="cpu",
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        self.device = device
        self.semantic_sensor = semantic_sensor
        self.voxel_map = voxel_map
        self.visualize_planner = visualize_planner
        self.cache_dir = cache_dir

    def reset(self):
        self.voxel_map.reset()

    ##############################################
    # Add new observations
    ##############################################
    def step(self, obs: Observations, visualize_map=False):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        device = self.device

        if self.semantic_sensor is not None:
            # This is slow because it gets passed back + forth to the CPU
            # And is done one-at-a-time.
            # Would be good to try caching these
            obs_for_semantic_sensor = Observations(
                rgb=obs.rgb.cpu().numpy(), gps=None, compass=None, depth=None
            )
            res = self.semantic_sensor.predict(
                obs_for_semantic_sensor
            ).task_observations
            instance_image = torch.from_numpy(res["instance_map"]).int().to(device)
            instance_classes = (
                torch.from_numpy(res["instance_classes"]).int().to(device)
            )
            instance_scores = (
                torch.from_numpy(res["instance_scores"]).float().to(device)
            )
            # semantic_frame = torch.from_numpy(res['semantic_frame']
        else:
            instance_image = obs.instance
            instance_classes = obs.task_observations["instance_classes"]
            instance_scores = obs.task_observations["instance_scores"]

        self.voxel_map.add(
            rgb=obs.rgb.float() / 255.0,
            depth=obs.depth.squeeze(-1),
            feats=obs.task_observations.get("features", None),
            camera_K=obs.camera_K,
            camera_pose=obs.camera_pose,  # scene_obs['axis_align_mats'][i] @ scene_obs['poses'][i],
            instance_image=instance_image,
            instance_classes=instance_classes,
            instance_scores=instance_scores,
        )

        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

    def step_trajectory(
        self, obs_list: Sequence[Observations], cache_key: Optional[str] = None
    ):
        """Tkes a list of observations and adds them all to the instance map"""
        if cache_key is not None:
            # load from cache
            assert self.cache_dir is not None
            raise NotImplementedError
        else:
            for i, obs in enumerate(obs_list):
                self.step(obs)
            if self.cache_dir is not None:
                # Save to cache
                raise NotImplementedError
        logger.debug(f"Found {len(self.voxel_map.get_instances())} instances")

    ##############################################
    # Language queries that return instances
    ##############################################
    def set_vocabulary(self, vocabulary: Dict[int, str]):
        vocabulary = RearrangeDETICCategories(vocabulary)
        self.semantic_sensor.update_vocabulary_list(vocabulary, SemanticVocab.SIMPLE)
        self.semantic_sensor.set_vocabulary(SemanticVocab.SIMPLE)
        return self.semantic_sensor.current_vocabulary

    def get_instances_for_query(self, text_query: str) -> List[Instance]:
        assert (
            text_query in self.semantic_sensor.name_to_seg_id
        ), f"{text_query} not in semantic_sensor vocabulary (current vocab: {self.semantic_sensor.current_vocabulary_id})"
        query_class_id = self.semantic_sensor.name_to_seg_id[text_query]
        instances = self.voxel_map.get_instances()
        instances = [inst for inst in instances if inst.category_id == query_class_id]
        return instances

    def build_scene_and_get_instances_for_queries(
        self, scene_obs: Dict[str, Any], queries: Sequence[str]
    ) -> Dict[str, List[Instance]]:
        """_summary_

        Args:
            scene_obs (Dict[str, Any]): Contains
                - Images
                - Depths
                - Poses
                - Intrinsics
                - scan_name -- str that could be used for caching (but we probably also want to pass in dataset or sth in case we change resoluton, frame_skip, etc)
            queries (Sequence[str]): Text queries, processed independently

        Returns:
            Dict[str, List[Instance]]: mapping queries to instances
        """
        # Build scene representation
        obs_list = []
        for i in range(len(scene_obs["images"])):
            obs = Observations(
                gps=None,
                compass=None,
                rgb=scene_obs["images"][i] * 255,
                depth=scene_obs["depths"][i],
                semantic=None,
                instance=None,  # These could be cached
                # Pose of the camera in world coordinates
                camera_pose=scene_obs["poses"][i],
                camera_K=scene_obs["intrinsics"][i],
                task_observations={
                    # "features": scene_obs["images"][i],
                },
            )
            obs_list.append(obs)
        self.step_trajectory(obs_list)
        self.voxel_map.postprocess_instances()

        # Get query results
        instances_dict = {}
        for class_name in queries:
            instances_dict[class_name] = self.get_instances_for_query(class_name)
        self.reset()
        return instances_dict

    ##############################################
    # 2D map projections for planning
    ##############################################
    def get_2d_map(self):
        """Get 2d obstacle map for low level motion planning and frontier-based exploration"""
        return self.voxel_map.get_2d_map()

    def show(self) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        return self.voxel_map.show(
            instances=True,
            height=1000,
            boxes_plot_together=False,
            boxes_name_int_to_display_name_dict=dict(
                enumerate(self.metadata.thing_classes)
            ),
            backend="pytorch3d",
        )


@store(name="main")
@torch.no_grad()
def main(
    model: SparseVoxelMapAgent,
    dataset: ScanNetDataset,
    scene_name: str = "scene0192_00",
    show_backend: Optional[str] = "open3d",
    torch_device: str = "cuda",
):
    """Builds a SparseVoxelMap for a ScanNetScene and evaluates object detection against GT

    Args:
        model (SparseVoxelMapAgent): _description_
        dataset (ScanNetDataset): _description_
        scene_name (str, optional): _description_. Defaults to 'scene0192_00'.
        show_backend (Optional[str], optional): _description_. Defaults to 'open3d'.
        torch_device (str, optional): _description_. Defaults to 'cuda'.
    """
    idx = dataset.scene_list.index(scene_name)
    scene_obs = dataset.__getitem__(idx, show_progress=True)
    # Move other keys to device
    for key in ["images", "depths", "intrinsics", "poses"]:
        scene_obs[key] = scene_obs[key].to(torch_device)

    # Add to ScanNetSparseVoxelMap
    n_frames = len(scene_obs["images"])
    with torch.no_grad():
        for i in tqdm(range(n_frames), desc="Adding observations to map"):
            obs = Observations(
                gps=None,
                compass=None,
                rgb=scene_obs["images"][i] * 255,
                depth=scene_obs["depths"][i],
                semantic=None,
                instance=scene_obs["instance_map"][i],
                # Pose of the camera in world coordinates
                camera_pose=scene_obs["poses"][i],
                camera_K=scene_obs["intrinsics"][i],
                task_observations={
                    # "instance_classes": scene_obs["instance_classes"][i],
                    # "instance_scores": scene_obs["instance_scores"][i],
                    # "features": scene_obs["images"][i],
                },
            )
            model.step(obs)

    # Evaluate against GT object detection + localization
    instances = model.voxel_map.get_instances()
    bounds = torch.stack([inst.bounds.cpu() for inst in instances])
    scores = torch.stack(
        [
            torch.mean(torch.stack([v.score for v in ins.instance_views])).cpu()
            for ins in instances
        ]
    )
    classes = torch.stack([inst.category_id.cpu() for inst in instances])
    detic_to_scannet_classes = torch.tensor(dataset.METAINFO["CLASS_IDS"])
    classes = detic_to_scannet_classes[classes]

    eval_bboxes_and_print(
        box_gt_bounds=[scene_obs["boxes_aligned"]],
        box_gt_class=[scene_obs["box_classes"]],
        box_pred_bounds=[bounds],
        box_pred_class=[classes],
        box_pred_scores=[scores],
        match_within_class=True,
        iou_thr=(0.25, 0.5, 0.75),
        # label_to_cat=
    )

    if show_backend:
        model.voxel_map.show(backend=show_backend)


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("default")
    store.add_to_hydra_store()
    zen(main).hydra_main(
        version_base="1.3",
        config_name="eval.yaml",
        config_path="configs",
    )
