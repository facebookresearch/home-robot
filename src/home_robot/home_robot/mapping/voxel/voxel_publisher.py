# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pickle
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from atomicwrites import atomic_write
from tqdm import tqdm

from home_robot.core.interfaces import Observations
from home_robot.mapping.semantic.instance_tracking_modules import Instance
from home_robot.mapping.voxel.voxel import SparseVoxelMap
from home_robot.utils.threading import Interval

logger = logging.getLogger(__name__)


def publish_obs(model: SparseVoxelMap, publish_dir: Path, timestep: int):
    with atomic_write(publish_dir / f"{timestep}.pkl", mode="wb") as f:
        model_obs = model.voxel_map.observations[timestep]

        instances = model.voxel_map.get_instances()
        if len(instances) > 0:
            bounds, names = zip(*[(v.bounds, v.category_id) for v in instances])
            bounds = torch.stack(bounds, dim=0)
            names = torch.stack(names, dim=0).unsqueeze(-1)
        else:
            bounds = torch.zeros(0, 3, 2)
            names = torch.zeros(0, 1)
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
                box_bounds=bounds,
                box_names=names,
            ),
            f,
        )


class FilePublisher:
    """
    A quick + dirty class that builds up a SparseVoxelMap from obserations,
    and publishes them to a directory, with `fps` publishes per second.
    """

    def __init__(
        self,
    ):
        self.current_obs = 0

    def _publish(
        self,
        model: SparseVoxelMap,
        publish_dir: Path,
        observations: Observations,
        progress_callback=None,
    ) -> bool:
        if self.current_obs >= len(observations):
            return False
        i = self.current_obs
        obs = observations[i]
        model.step(obs)
        publish_obs(model=model, publish_dir=publish_dir, timestep=i)
        self.current_obs += 1
        if progress_callback is not None:
            progress_callback(1)
        return self.current_obs < len(observations)

    def build_representation_and_publish(
        self,
        publish_dir: Path,
        model: SparseVoxelMap,
        observations: Observations,
        wait_keypress_for_start: bool = False,
        wait_keypress_for_continue: bool = False,
        fps: Optional[int] = None,
    ):
        self.current_obs = 0
        dump_dir = Path(publish_dir)
        assert dump_dir and str(dump_dir) not in [".", "/"]
        dump_dir = dump_dir.resolve()
        obs_dir = dump_dir / "obs"

        if fps is None or fps == "inf" or fps <= 0:
            write_interval = 0.0
            fps = "inf"
        else:
            write_interval = 1.0 / fps
        if wait_keypress_for_continue:
            write_interval = None
        if write_interval is None:
            logger.info("Waiting for keypress to continue each write")
        else:
            logger.info("Writing obs every {write_interval:.2f} sec")

        start_time = time.time()

        n_obs = len(observations)
        desc = f"{fps}" if write_interval is not None else "Press 'enter' to continue"
        with tqdm(total=n_obs, desc=desc) as t:

            timer = Interval(
                partial(
                    self._publish,
                    model=model,
                    publish_dir=obs_dir,
                    observations=observations,
                    progress_callback=t.update,
                ),
                sleep_time=write_interval,
            )

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
                    _ = input(
                        f"\nPress enter to continue ({len(os.listdir(obs_dir))}):"
                    )
                    timer.unpause()
                    time.sleep(0.1)
            timer.join()

        end_time = time.time()
        logger.info(
            f"Wrote {n_obs} frames at {n_obs / (end_time - start_time):0.2f} FPS (target {fps} FPS)"
        )
