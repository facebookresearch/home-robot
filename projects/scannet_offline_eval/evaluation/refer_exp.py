# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    This file contains evaluation utilities for referring expressions.
    I.e. "The X closer to the Y" that indicate a specific object
    Metrics are proportion of IoUs above a threshold
    We can measure this for
        - identifying the correct object
        - IDing a different object of the target's class
        - Identfying a different object
    For each of these we can also measure average IoU given that it was above
        a threshold.
"""
import warnings
from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import pytorch3d
import torch
from pytorch3d.ops import box3d_overlap
from terminaltables import AsciiTable
from torch import Tensor

from home_robot.utils.bboxes_3d import get_box_verts_from_bounds

# Return:
# - IoU and IoU @ thresh
# - Against target
# - Against target class (is it getting distracted by other objects of the same class)
# - Against all objects (is it localizing any object?)


@torch.no_grad()
def get_stats_ious_at_thresh(
    ious: Tensor,
    iou_thresh: Sequence[float],
    is_unique: Optional[Tensor],
):
    """Compute summary statistics given a list of ious and thesholds at which to evaluate

    Args:
        ious (Tensor): iou results for a set of bboxes/masks
        iou_thresh (Sequence[float]): list of iou thresholds for which to evaluate summary stats

    Returns:
        prop_at_iou (Dict[float, float]): Mapping iou_thresh to proportion of ious > thresh
        mean_iou_given_thresh (Dict[float, float]): Mean IoU | IoU > iou_thresh
        prop_at_iou_unique (Dict[float, float]): Mapping iou_thresh to proportion of ious > thresh
        mean_iou_given_thresh_unique (Dict[float, float]): Mean IoU | IoU > iou_thresh
        prop_at_iou_multiple (Dict[float, float]): Mapping iou_thresh to proportion of ious > thresh
        mean_iou_given_thresh_multiple (Dict[float, float]): Mean IoU | IoU > iou_thresh

        mean_iou (float): Average IoU
    """
    results = defaultdict({})
    for iou in iou_thresh:
        is_above_thresh = ious > iou
        ious_above_thresh = ious[is_above_thresh]
        results["prop_at_iou_overall"][
            iou
        ] = is_above_thresh.mean()  # On a bool tensor returns float
        results["mean_iou_given_thresh_overall"][iou] = ious[is_above_thresh].mean()
        if is_unique is not None:
            results["prop_at_iou_unique"][iou] = is_above_thresh[is_unique].mean()
            results["mean_iou_given_thresh_unique"][iou] = ious[
                is_above_thresh & is_unique
            ].mean()

            results["prop_at_iou_multiple"][iou] = is_above_thresh[~is_unique].mean()
            results["mean_iou_given_thresh_multiple"][iou] = ious[
                is_above_thresh & (~is_unique)
            ].mean()
    results["mean_iou"] = ious.mean()
    return results


@torch.no_grad()
def eval_obj_selection_bboxes(
    box_gt_bounds: Sequence[Tensor],
    box_gt_class: Sequence[Tensor],
    box_gt_ids: Sequence[Tensor],
    exp_target_ids: Sequence[int],
    box_pred_bounds: Sequence[Tensor],
    iou_thr: Sequence[float] = (0.25, 0.5, 0.75),
):
    """
    Metrics are proportion of IoUs above a threshold
    We can measure this for
        - identifying the correct object
        - IDing a different object of the target's class
        - Identfying a different object
    For each of these we can also measure average IoU given that it was above
        a threshold.

    Also breaks down metrics by scenes where the object is the only object
        of the target class (unique), or there are distractors of the
        same class (multiple)

    Args:

    Returns:
        Dict[str, float]
        IoU of the prediction against all of the following:
        - Target bounding box (i.e. box_gt_bounds[i][target_ids[i]])
        - Other bounding boxes of the same class as the target (i.e. distractors)
            - This means the model isn't encoding the context and is just class-matching the prompt
        - Any bounding box in the scene. This disambiguates whether
            - The model mis-identifies the object (this IoU will be high)
            - The model is failing to localize (this will be low
    """
    classes = torch.cat(box_gt_class, dim=0).unique()
    device = classes.device
    cls_to_retvals = defaultdict(list)
    n_scenes = len(box_gt_bounds)
    assert len(box_gt_class) == n_scenes, len(box_gt_class)
    assert len(box_gt_ids) == n_scenes, len(box_gt_ids)
    assert len(exp_target_ids) == n_scenes, len(exp_target_ids)
    assert len(box_pred_bounds) == n_scenes, len(box_pred_bounds)

    # Compute summary statistics
    ious_targets = []
    ious_classes = []
    ious_anything = []
    unique_classes = []  # Whether this was the only instance of that class
    # Loop over each prediction, target_bbox_id, all_bboxes_in_scene
    for i, (
        _gt_bounds,
        _gt_class,
        _gt_obj_ids,
        _exp_target_ids,
        _pred_bounds,
    ) in enumerate(
        zip(
            box_gt_bounds,
            box_gt_class,
            box_gt_ids,
            exp_target_ids,
            box_pred_bounds,
        )
    ):
        assert len(_pred_bounds) == 1, _pred_bounds.shape
        assert len(_gt_bounds) == 1, _gt_bounds.shape

        _pred_corners = get_box_verts_from_bounds(_pred_bounds)
        _gt_corners = get_box_verts_from_bounds(_gt_bounds)
        _, box_iou = box3d_overlap(_pred_corners, _gt_corners)
        box_iou = box_iou[
            0
        ]  # We are only evaluating one detection at a time       target_iou = box_iou[o]

        # Get IoU against target
        assert (_gt_obj_ids == _exp_target_ids).sum() == 1, _gt_obj_ids.sum()
        target_iou = box_iou[_gt_obj_ids == _exp_target_ids]
        ious_targets.append(target_iou)

        # Get IoU against anything in the target class
        same_class = _gt_class == _exp_target_ids
        _gt_class[same_class]
        _gt_corners[same_class]
        class_iou = box_iou[same_class].max()
        ious_classes.append(class_iou)

        unique_class = same_class.sum() > 1
        unique_classes.append(unique_class)
        # Get IoU against anything
        anything_iou = box_iou.max()
        ious_anything.append(anything_iou)

    iou_types = [
        "target",
        "same_class",
        "anything",
    ]  # could have objects/receptacles too
    eval_dict = {}
    for iou_type, ious in zip(iou_types, [ious_targets, ious_classes, ious_anything]):
        eval_dict[iou_type] = get_stats_ious_at_thresh(
            ious, iou_thresh=iou_thr, unique_classes=torch.stack(unique_classes)
        )

    header = ["classes"]
    table_columns = []
    return_dict = {}

    metric_name = "mean_iou"
    header.append(metric_name)
    table_columns.append([eval_dict[iou_type] for iou_type in iou_types])
    return_dict[metric_name] = {
        iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
    }

    for i, iou in enumerate(iou_thr):
        metric_name = f"prop_gt_{iou:2f}_oval"
        header.append(metric_name)
        table_columns.append(
            [eval_dict[iou_type]["prop_at_iou_overall"][iou] for iou_type in iou_types]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

        metric_name = f"mean_iou_given_{iou:2f}_oval"
        header.append(metric_name)
        table_columns.append(
            [
                eval_dict[iou_type]["mean_iou_given_thresh_overall"][iou]
                for iou_type in iou_types
            ]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

        metric_name = f"prop_gt_{iou:2f}_uniq"
        header.append(metric_name)
        table_columns.append(
            [eval_dict[iou_type]["prop_at_iou_unique"][iou] for iou_type in iou_types]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

        metric_name = f"mean_iou_given_{iou:2f}_uniq"
        header.append(metric_name)
        table_columns.append(
            [
                eval_dict[iou_type]["mean_iou_given_thresh_unique"][iou]
                for iou_type in iou_types
            ]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

        metric_name = f"prop_gt_{iou:2f}_mult"
        header.append(metric_name)
        table_columns.append(
            [eval_dict[iou_type]["prop_at_iou_ultiple"][iou] for iou_type in iou_types]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

        metric_name = f"mean_iou_given_{iou:2f}_mult"
        header.append(metric_name)
        table_columns.append(
            [
                eval_dict[iou_type]["mean_iou_given_thresh_multiple"][iou]
                for iou_type in iou_types
            ]
        )
        return_dict[metric_name] = {
            iou_type: vals for iou_type, vals in zip(iou_types, table_columns[-1])
        }

    table_data = [header]
    table_rows = list(zip(*table_rows))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print("\n" + table.table)
    return eval_dict
