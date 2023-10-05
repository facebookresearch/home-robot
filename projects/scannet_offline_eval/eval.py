# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    Evaluates a model's ability to return correct bounding boxes given a text query
"""
import gc
from enum import IntEnum, auto
from typing import Callable, Sequence

# import build_sparse_voxel_map
import evaluation
import torch
from hydra_zen import builds, store, zen
from torch import Tensor
from tqdm import tqdm

import home_robot
from home_robot.core.interfaces import Observations


class InstanceModel:
    def step_traj(self, obs_list: Sequence[Observations]):
        raise NotImplementedError

    def get_instances_for_query(self, text: str):
        raise NotImplementedError


# 1) `hydra_zen.store generates a config for our task function
#    and stores it locally under the entry-name "my_app"
@store(name="eval_runner")
@torch.no_grad()
def eval_runner(
    model: InstanceModel, dataset: torch.utils.data.Dataset, eval_fn: Callable
):

    # SETUP EVAL
    class_id_to_class_names = dict(
        zip(
            dataset.METAINFO["CLASS_IDS"],  # IDs [1, 3, 4, 5, ..., 65]
            dataset.METAINFO["CLASS_NAMES"],  # [wall, floor, cabinet, ...]
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
        gc.collect()  # Help prevent OOM errors

        # Move to device
        for k in keys:
            scene_obs[k] = scene_obs[k].to(model.device)

        # Eval each scene and move to CPU
        queries = {
            int(clas): class_id_to_class_names[int(clas)]
            for clas in scene_obs["box_classes"].unique()
        }
        instances_dict = model.build_scene_and_get_instances_for_queries(
            scene_obs, queries.values()
        )
        (
            scene_gt_bounds,
            scene_gt_classes,
            scene_pred_bounds,
            scene_pred_classes,
            scene_pred_scores,
        ) = ([], [], [], [], [])
        for clas, class_name in queries.items():
            gt_class_match = scene_obs["box_classes"] == int(clas)
            if len(gt_class_match) == 0:
                raise RuntimeError(
                    f"No GT for class {class_name} in {scene_obs['scan_name']}"
                )
            scene_gt_bounds.append(scene_obs["boxes_aligned"][gt_class_match].cpu())
            scene_gt_classes.append(scene_obs["box_classes"][gt_class_match].cpu())
            instances = instances_dict[class_name]
            if len(instances) > 0:
                _class_pred_bounds = (
                    torch.stack([inst.bounds for inst in instances], dim=0)
                    .detach()
                    .cpu()
                )
                _class_pred_class = torch.full(
                    (len(instances),), int(clas), dtype=scene_obs["box_classes"].dtype
                )
                _class_pred_scores = (
                    torch.stack([inst.score for inst in instances]).detach().cpu()
                )
            else:
                _class_pred_bounds = torch.zeros(
                    [0, 3, 2], dtype=scene_obs["boxes_aligned"].dtype
                )
                _class_pred_class = torch.zeros(
                    [
                        0,
                    ],
                    dtype=scene_obs["box_classes"].dtype,
                )
                _class_pred_scores = torch.zeros(
                    [
                        0,
                    ],
                    dtype=scene_obs["boxes_aligned"].dtype,
                )
            scene_pred_bounds.append(_class_pred_bounds)
            scene_pred_classes.append(_class_pred_class)
            scene_pred_scores.append(_class_pred_scores)

        # Aggregate on CPU and evaluate
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
    import warnings

    warnings.simplefilter("default")
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
