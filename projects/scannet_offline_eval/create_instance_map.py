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

from home_robot.agent.multitask import BaseMultiTaskAgent
from home_robot.core.interfaces import Observations
from home_robot.datasets.scannet import ScanNetDataset
from home_robot.mapping.semantic.instance_tracking_modules import Instance

logger = logging.getLogger(__name__)


@store(name="main")
@torch.no_grad()
def main(
    model: BaseMultiTaskAgent,
    dataset: ScanNetDataset,
    scene_name: str = "scene0192_00",
    show_backend: Optional[str] = "open3d",
    torch_device: str = "cuda",
    save_instance_map_fpath: Optional[Union[Path, str]] = None,
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
    model.build_scene_and_get_instances_for_queries(scene_obs, queries=[], reset=False)

    # Evaluate against GT object detection + localization
    instances = model.voxel_map.get_instances()
    logger.info(f"Found {len(instances)} instances")
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

    if save_instance_map_fpath:
        torch.save(model, save_instance_map_fpath)


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("default")
    store.add_to_hydra_store()
    zen(main).hydra_main(
        version_base="1.3",
        config_name="build_sparse_voxel_map.yaml",
        config_path="configs",
    )
