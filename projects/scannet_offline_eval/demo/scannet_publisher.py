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
from timer import TimerClass
from torch import Tensor
from tqdm import tqdm

from home_robot.agent.multitask.sparse_voxel_instance_map import SparseVoxelMapAgent
from home_robot.core.interfaces import Observations
from home_robot.datasets.scannet import ScanNetDataset
from home_robot.mapping.semantic.instance_tracking_modules import Instance

logger = logging.getLogger(__name__)


class Publisher:
    def __init__(
        self,
        model,
        scene_obs,
        publish_dir: Path,
        progress_callback: Optional[Callable] = None,
    ):
        self.current_obs = 0
        self.publish_dir = publish_dir
        self.n_obs = len(scene_obs["images"])
        self.scene_obs = scene_obs
        self.model = model
        self.progress_callback = progress_callback

    def publish(self) -> bool:
        if self.current_obs >= self.n_obs:
            return False
        i = self.current_obs
        scene_obs = self.scene_obs
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
        self.model.step(obs)
        # obs_list.append(obs)

        with atomic_write(self.publish_dir / f"{i}.pkl", mode="wb") as f:
            model_obs = self.model.voxel_map.observations[i]
            pickle.dump(
                dict(
                    rgb=model_obs.rgb.cpu().detach(),
                    depth=model_obs.depth.cpu().detach(),
                    instance_image=model_obs.instance.cpu().detach(),
                    instance_classes=model_obs.instance_classes.cpu().detach(),
                    instance_scores=model_obs.instance_scores.cpu().detach(),
                    camera_pose=model_obs.camera_pose.cpu().detach(),
                    camera_K=model_obs.camera_K.cpu().detach(),
                    xyz_frame=model_obs.xyz_frame,
                ),
                f,
            )
        self.current_obs += 1
        if self.progress_callback is not None:
            self.progress_callback(1)
        return self.current_obs < self.n_obs


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
    dump_dir = Path(dump_dir)
    assert dump_dir and str(dump_dir) not in [".", "/"]
    dump_dir = dump_dir.resolve()
    obs_dir = dump_dir / "obs"

    write_interval = 1.0 / fps
    if wait_keypress_for_continue:
        write_interval = None
    if write_interval is None:
        logger.info("Waiting for keypress to continue each write")
    else:
        logger.info("Writing obs every {write_interval:.2f} sec")

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

    start_time = time.time()

    n_obs = len(scene_obs["images"])
    desc = f"{fps}" if write_interval is not None else "Press 'enter' to continue"
    with tqdm(total=len(scene_obs["images"]), desc=desc) as t:
        publisher = Publisher(
            model, scene_obs, publish_dir=obs_dir, progress_callback=t.update
        )
        timer = TimerClass(publisher.publish, sleep_time=write_interval)

        # Setup
        logger.info(f"\nClearing and writing to {dump_dir}")
        if wait_keypress_for_start:
            _ = input("\nPress enter to start:")
        if os.path.exists(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(obs_dir)
        timer.start()

        # Maybe wait for user input
        if wait_keypress_for_continue:
            while not timer.event.is_set():
                _ = input(f"\nPress enter to continue ({len(os.listdir(obs_dir))}):")
                timer.unpause()
                time.sleep(0.1)
        timer.join()

    end_time = time.time()
    logger.info(
        f"Wrote {n_obs} frames at {n_obs / (end_time - start_time):0.2f} FPS (target {fps} FPS)"
    )


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("default")
    store.add_to_hydra_store()
    zen(main).hydra_main(
        version_base="1.3",
        config_name="scannet_publisher.yaml",
        config_path="../configs",
    )
