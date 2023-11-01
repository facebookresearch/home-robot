# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, Union

import plotly.graph_objects as go
import torch
from atomicwrites import atomic_write
from hydra_zen import builds, store, zen
from torch import Tensor
from tqdm import tqdm

from home_robot.agent.multitask.sparse_voxel_instance_map import SparseVoxelMapAgent
from home_robot.core.interfaces import Observations
from home_robot.datasets.scannet import ScanNetDataset
from home_robot.mapping.semantic.instance_tracking_modules import Instance
from home_robot.mapping.voxel.voxel_publisher import FilePublisher

logger = logging.getLogger(__name__)


@store(name="main")
@torch.no_grad()
def main(
    model: SparseVoxelMapAgent,
    dataset: ScanNetDataset,
    trajectory_pickle_path: Union[Path, str],
    dump_dir: Union[Path, str],
    scene_name: str = "scene0192_00",
    torch_device: str = "cuda",
    fps: int = 1,
    wait_keypress_for_start: bool = False,
    wait_keypress_for_continue: bool = False,
):
    """Builds a SparseVoxelMap for a ScanNetScene and evaluates object detection against GT

    Args:
        model (SparseVoxelMapAgent): _description_
        dataset (ScanNetDataset): _description_
        scene_name (str, optional): _description_. Defaults to 'scene0192_00'.
        show_backend (Optional[str], optional): _description_. Defaults to 'open3d'.
        torch_device (str, optional): _description_. Defaults to 'cuda'.
    """
    print(dataset.scene_list)

    class_id_to_class_names = dict(
        zip(
            dataset.METAINFO["CLASS_IDS"],  # IDs [1, 3, 4, 5, ..., 65]
            dataset.METAINFO["CLASS_NAMES"],  # [wall, floor, cabinet, ...]
        )
    )
    model.set_vocabulary(class_id_to_class_names)
    with open(trajectory_pickle_path, "rb") as f:
        observations = pickle.load(f)["obs"]
    for obs in observations:
        obs.rgb = obs.rgb.to(torch_device)
        obs.depth = obs.depth.to(torch_device)
        obs.camera_pose = obs.camera_pose.to(torch_device)
        obs.camera_K = torch.from_numpy(obs.camera_K).float().to(torch_device)

        # obs.task_observations['instance_map'] = torch.from_numpy(obs.task_observations['instance_map']).to(device)
        # obs.task_observations['instance_map'] = torch.from_numpy(obs.task_observations['instance_map']).to(device)
        # obs.task_observations['instance_map'] = torch.from_numpy(obs.task_observations['instance_map']).to(device)

    publisher = FilePublisher()
    publisher.build_representation_and_publish(
        publish_dir=dump_dir,
        model=model,
        observations=observations,
        wait_keypress_for_start=wait_keypress_for_start,
        wait_keypress_for_continue=wait_keypress_for_continue,
    )


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("default")
    store.add_to_hydra_store()
    zen(main).hydra_main(
        version_base="1.3",
        config_name="canned_robot_publisher.yaml",
        config_path="../../configs",
    )
