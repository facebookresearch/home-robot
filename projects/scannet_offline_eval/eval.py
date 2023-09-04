# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    Evaluates a model's ability to return correct bounding boxes given a text query
"""
from enum import IntEnum, auto
from typing import Callable, Sequence

import build_sparse_voxel_map
import evaluation
import torch
from hydra_zen import builds, store, zen
from torch import Tensor
from tqdm import tqdm

from home_robot.core.interfaces import Observations


class InstanceModel:
    def step_traj(self, obs_list: Sequence[Observations]):
        raise NotImplementedError

    def get_instances_for_query(self, text: str):
        raise NotImplementedError


# 1) `hydra_zen.store generates a config for our task function
#    and stores it locally under the entry-name "my_app"
@store(name="eval_runner")
def eval_runner(
    model: InstanceModel, dataset: torch.utils.data.Dataset, eval_fn: Callable
):
    class_id_to_class_names = dict(
        zip(
            dataset.METAINFO["seg_valid_class_ids"],  # IDs [1, 3, 4, 5, ..., 65]
            dataset.METAINFO["classes"],  # [wall, floor, cabinet, ...]
        )
    )
    # If this is an open-vocab detector, they sometimes require a vocab
    model.set_vocabulary(class_id_to_class_names)

    keys = [
        "images",
        "depths",
        "poses",
        "intrinsics",
        # 'instance_map',
        # 'instance_scores',
        # 'instance_classes',
        "boxes_aligned",
        "box_classes",
        # Include pixel embeddings here?
    ]

    # Evaluate on all scenes one at a time
    # TODO: Could make this multi-gpu to speed things up
    gt_bounds, gt_classes, pred_bounds, pred_classes, pred_scores = [], [], [], [], []
    for scene_obs in tqdm(dataset, desc="Evaluating scenes..."):
        # Garbage collect at beginning to free RAM/Device mem?

        # Move to device
        for k in keys:
            scene_obs[k] = scene_obs[k].to(model.device)

        # Build scene representation
        obs_list = []
        for i in range(len(scene_obs["images"])):
            obs = Observations(
                gps=None,
                compass=None,
                rgb=scene_obs["images"][i] * 255,
                depth=scene_obs["depths"][i],
                semantic=None,
                # Instance IDs per observation frame
                # Size: (camera_height, camera_width)
                # Range: 0 to max int
                instance=None,
                # Pose of the camera in world coordinates
                camera_pose=scene_obs["poses"][i],
                camera_K=scene_obs["intrinsics"][i],
                task_observations={
                    # "features": scene_obs["images"][i],
                },
            )
            obs_list.append(obs)
        model.step_trajectory(obs_list)

        # Eval this scene
        data_classes = scene_obs["box_classes"].unique()
        device = data_classes.device
        dtype = data_classes.dtype

        (
            scene_gt_bounds,
            scene_gt_classes,
            scene_pred_bounds,
            scene_pred_classes,
            scene_pred_scores,
        ) = ([], [], [], [], [])
        for clas in data_classes:
            class_name = class_id_to_class_names[int(clas)]
            scene_gt_bounds.append(scene_obs["boxes_aligned"])
            scene_gt_classes.append(scene_obs["box_classes"])
            instances = model.get_instances_for_query(class_name)
            scene_pred_bounds.append(
                torch.stack([inst.bounds for inst in instances], dim=0)
            )
            scene_pred_classes.append(
                torch.full_like(len(instances), int(clas), dtype=dtype, device=device)
            )
            scene_pred_scores.append(torch.stack([inst.score for inst in instances]))

        for combined_list, scene_results in zip(
            [gt_bounds, gt_classes, pred_bounds, pred_classes, pred_scores],
            [
                scene_gt_bounds,
                scene_gt_classes,
                scene_pred_bounds,
                scene_pred_classes,
                scene_pred_scores,
            ],
        ):
            combined_list.append(torch.cat(scene_results, dim=0).cpu().detach())

    # Get metrics and log
    result_dict = eval_fn(
        box_gt_bounds=gt_bounds,
        box_gt_class=gt_classes,
        box_pred_bounds=pred_bounds,
        box_pred_class=pred_classes,
        box_pred_scores=pred_scores,
        label_to_cat=class_id_to_class_names,
    )


# 2) Executing `python eval.py [...]` will run eval_runner
if __name__ == "__main__":
    # 3) We need to add the configs from our local store to Hydra's
    #    global config store
    store.add_to_hydra_store()

    # 4) Our zen-wrapped eval_runner is used to generate
    #    the CLI, and to specify which config we want to use
    #    to configure the app by default
    zen(eval_runner).hydra_main(
        version_base="1.3",
        config_name="eval.yaml",
        config_path="configs",
    )
