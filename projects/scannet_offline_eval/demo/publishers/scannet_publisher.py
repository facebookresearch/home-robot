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
    idx = dataset.scene_list.index(scene_name)
    scene_obs = dataset.__getitem__(idx, show_progress=True)
    # Move other keys to device
    for key in ["images", "depths", "intrinsics", "poses"]:
        scene_obs[key] = scene_obs[key].to(torch_device)

    class_id_to_class_names = dict(
        zip(
            dataset.METAINFO["CLASS_IDS"],  # IDs [1, 3, 4, 5, ..., 65]
            dataset.METAINFO["CLASS_NAMES"],  # [wall, floor, cabinet, ...]
        )
    )
    model.set_vocabulary(class_id_to_class_names)

    observations = [
        Observations(
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
        for i in range(len(scene_obs["images"]))
    ]
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
        config_name="scannet_publisher.yaml",
        config_path="../../configs",
    )
