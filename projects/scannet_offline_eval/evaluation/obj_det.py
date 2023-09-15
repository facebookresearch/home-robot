# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from collections import defaultdict
from typing import Dict, Sequence

import numpy as np
import pytorch3d
import torch
from pytorch3d.ops import box3d_overlap
from terminaltables import AsciiTable
from torch import Tensor

from home_robot.utils.bboxes_3d import (
    box3d_intersection_from_bounds,
    get_box_verts_from_bounds,
)


def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision (for single or multiple scales).
    # Copied from https://mmdetection3d.readthedocs.io/en/v0.15.0/_modules/mmdet3d/core/evaluation/indoor_eval.html
    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets) \
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of \
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    return ap


@torch.no_grad()
def get_det_assigments_to_gt(
    box_gt_bounds: Tensor,
    box_gt_class: Tensor,
    box_pred_bounds: Tensor,
    box_pred_class: Tensor,
    box_pred_scores: Tensor,
    iou_thr: Sequence[float] = (0.25, 0.5, 0.75),
    ap_mode: str = "area",
    eps: float = 1e-6,
):
    """_summary_

    Args:
        box_gt_bounds (Tensor): Ground truth box (N_gt, 3, 2)
        box_gt_class (Tensor): Ground truth box class (N_gt,)
        box_pred_bounds (Tensor): Predicted boxes (N_det, 3, 2)
        box_pred_class (Tensor): Predicted boxes classes (N_det,)
        box_pred_scores (Tensor): Predicted boxes detection scores (N_det,)
        iou_thr (Sequence[float], optional): IOU thresholds to evaluate at. Defaults to (0.25, 0.5, 0.75).

    Returns:
        Dict:
            box_pred_sort_idx (longTensor): idxs to sort input box_preds by confidence
            box_pred_bounds (Tensor): bounds sorted by box_pred_sort_idx
            box_pred_scores (Tensor): scores sorted by box_pred_sort_idx
            box_pred_class (Tensor): classes sorted by box_pred_sort_idx
            is_true_pos (BoolTensor): If the assignment det_to_gt is a TP or FP
            det_to_gt (IntTensor): [N_For each detection, which GT is has the best match with
            ious (Tensor): For each match, the corresponding IoU
            gt_to_det (IntTensor): [N_iou_thr, N_gt] For each GT, which det is the best match. -1 for no match

    """
    n_gt = len(box_gt_bounds)
    assert n_gt > 0, "No GT labels"
    assert len(box_gt_class) == n_gt, box_gt_class.shape

    n_det = len(box_pred_bounds)
    assert len(box_pred_bounds) == n_det, box_pred_bounds.shape
    assert len(box_pred_class) == n_det, box_pred_class.shape
    assert len(box_pred_scores) == n_det, box_pred_scores.shape

    device = box_gt_class.device
    is_true_pos = torch.zeros(
        (len(iou_thr), n_det), dtype=torch.bool, device=device
    )  # 0: FP, +1 TP,
    gt_to_det_at_thr = torch.full(
        (len(iou_thr), n_gt), -1, device=device
    )  # idx of detection that is best match
    gt_matched_with_higher_conf = torch.full(
        (len(iou_thr), n_gt), False, dtype=torch.bool, device=device
    )  # N_thr, N_gt

    box_pred_sorted_idx = torch.argsort(box_pred_scores)
    box_pred_bounds = box_pred_bounds[box_pred_sorted_idx]
    box_pred_scores = box_pred_scores[box_pred_sorted_idx]
    box_pred_class = box_pred_class[box_pred_sorted_idx]

    if n_det == 0:
        return dict(
            box_pred_sorted_idx=box_pred_sorted_idx,
            box_pred_bounds=box_pred_bounds,
            box_pred_scores=box_pred_scores,
            box_pred_class=box_pred_class,
            is_true_pos=is_true_pos,
            det_to_gt=torch.full((n_det,), -1, dtype=torch.int, device=device),
            ious=torch.zeros_like(box_pred_scores),
            gt_to_det=gt_to_det_at_thr,
        )

    iou_thr = torch.tensor(iou_thr, device=box_gt_bounds.device)

    max_match_iou, matches = [], []
    # Go down the detections list in order of descending confidence, and associate to GT

    # # This code would work for oriented bounding boxes, but we would have to do more stringent checking for small bboxes
    # # Since box3d_overlap likes to throw errors when box3d volume < eps.
    # box_gt_corners = get_box_verts_from_bounds(box_gt_bounds).float()
    # box_pred_corners = get_box_verts_from_bounds(box_pred_bounds).float()
    # for i, cur_box_pred_corners in enumerate(box_pred_corners):
    #     # It's just too irritating to use box3d_overlap
    #     _, box_iou = box3d_overlap(
    #         cur_box_pred_corners.unsqueeze(0),
    #         box_gt_corners,
    #         eps,
    #     )  # (1, N_gt)

    # Instead, all our boxes are axis-aligned so we can just use this code which is probably slightly faster anyway.
    for i, cur_box_pred_bounds in enumerate(box_pred_bounds):
        _, box_iou, _ = box3d_intersection_from_bounds(
            cur_box_pred_bounds.unsqueeze(0), box_gt_bounds, eps
        )

        box_iou = box_iou[0]  # We are only evaluating one detection at a time
        max_iou, gt_match_idx = box_iou.max(dim=0)  # (N_gt)
        max_match_iou.append(max_iou)
        matches.append(gt_match_idx)

        is_f_pos = gt_matched_with_higher_conf[:, gt_match_idx] | (max_iou < iou_thr)
        is_true_pos[:, i] = ~is_f_pos
        gt_to_det_at_thr[:, gt_match_idx] = torch.where(
            is_true_pos[:, i], i, gt_to_det_at_thr[:, gt_match_idx]
        )
        gt_matched_with_higher_conf[:, gt_match_idx] = True

    assert torch.all(
        is_true_pos.sum(dim=-1) <= n_gt
    ), "Somehow classified more than 1 detection to the same GT bbox!"
    return dict(
        box_pred_sorted_idx=box_pred_sorted_idx,
        box_pred_bounds=box_pred_bounds,
        box_pred_scores=box_pred_scores,
        box_pred_class=box_pred_class,
        is_true_pos=is_true_pos,
        det_to_gt=torch.stack(matches),
        ious=torch.stack(max_match_iou),
        gt_to_det=gt_to_det_at_thr,
    )


@torch.no_grad()
def compute_ap_recall(
    is_true_pos: Sequence[Tensor],
    box_pred_scores: Sequence[Tensor],
    n_pos: Sequence[int],
    ap_mode: str = "area",
):
    """Compute AP and Recall given TP/FP boolean matrix and scores

    Args:
        is_true_pos (List[Tensor]): [N_scales, N_det] Per-scene list of tensors containing whether each detection is a true/false positive
        box_pred_scores (List[Tensor]): [N_det] Per-scene list of tensors contatinin detection scores. Used to rank detections for precision/recall
        n_pos (List[int]): How many GT positives there are per scene -- used to compute recall
        ap_mode (str, optional): How to compute average precision. Defaults to 'area'.

    Returns:
        recall (Tensor): [N_scales] recall
        ap (Tensor): [N_scales] average precision
        n_dets (int):
    """
    is_true_pos = torch.cat(is_true_pos, dim=1)  # N_iou_thr, N_dets_combined
    scores = torch.cat(box_pred_scores)  # N_dets_combined
    # Sort by confidence -- ranking neede for precision/recall
    sort_idxs = torch.argsort(scores, dim=-1, descending=True, stable=False)
    is_true_pos = is_true_pos[:, sort_idxs]
    n_pos = sum(n_pos)

    # Calculate precision and recall
    tp = torch.cumsum(is_true_pos, dim=-1)
    fp = torch.cumsum(~is_true_pos, dim=-1)
    recall = tp / float(n_pos)
    precision = tp / (tp + fp).float().clamp(
        min=1e-8
    )  # avoid zero div in case nothing matched the first GT
    ap = torch.tensor(
        average_precision(recall.numpy(), precision.numpy(), mode=ap_mode)
    )
    if recall.shape[-1] == 0:
        recall = torch.zeros_like(ap)
    else:
        recall = recall[:, -1]
    return ap, recall, precision.shape[-1]


@torch.no_grad()
def compute_box_det_ap_recall(
    box_gt_bounds: Sequence[Tensor],
    box_gt_class: Sequence[Tensor],
    box_pred_bounds: Sequence[Tensor],
    box_pred_class: Sequence[Tensor],
    box_pred_scores: Sequence[Tensor],
    match_within_class: bool = True,
    iou_thr: Sequence[float] = (0.25, 0.5, 0.75),
    ap_mode: str = "area",
    all_class: int = 0,
    eps: float = 1e-6,
):
    """Compute Average Precision (AP) and Recall for object detection tasks.

    This function calculates AP and Recall for each class, given the ground-truth and predicted bounding boxes, classes, and scores.

    Args:
        box_gt_bounds (Sequence[Tensor]): Ground-truth bounding boxes for each scene, represented as a sequence of PyTorch tensors.
        box_gt_class (Sequence[Tensor]): Ground-truth classes for each ground-truth bounding box, represented as a sequence of PyTorch tensors.
        box_pred_bounds (Sequence[Tensor]): Predicted bounding boxes for each scene, represented as a sequence of PyTorch tensors.
        box_pred_class (Sequence[Tensor]): Predicted classes for each predicted bounding box, represented as a sequence of PyTorch tensors.
        box_pred_scores (Sequence[Tensor]): Confidence scores for each predicted bounding box, represented as a sequence of PyTorch tensors.
        match_within_class (bool, optional): Whether to match boxes within the same class only. Defaults to True.
        iou_thr (Sequence[float], optional): Sequence of IoU thresholds to consider for calculating AP and Recall. Defaults to (0.25, 0.5, 0.75).
        ap_mode (str, optional): How to compute average precision. Defaults to 'area'.
        all_class (int): if match_within_class is false, what key in the return dicts to use for the superclass
    Returns:
        ap (dict): A dictionary containing average precision (AP) values for each class. The key is the class ID and the value is the AP.
        recall (dict): A dictionary containing recall values for each class. The key is the class ID and the value is the recall.

    Note:
        This function internally calls `get_det_assigments_to_gt()` to assign detections to ground-truth boxes based on IoU.
        The final AP and Recall are calculated per class and can be accessed in the returned dictionaries.
    """
    # classes = torch.cat(box_gt_class, dim=0).unique()
    cls_to_retvals = defaultdict(list)
    n_scenes = len(box_gt_bounds)
    assert len(box_gt_class) == n_scenes, len(box_gt_class)
    assert len(box_pred_bounds) == n_scenes, len(box_pred_bounds)
    assert len(box_pred_class) == n_scenes, len(box_pred_class)
    assert len(box_pred_scores) == n_scenes, len(box_pred_scores)

    # Evaluate each scene individually
    for i, (
        _gt_bounds,
        _gt_class,
        _pred_bounds,
        _pred_class,
        _pred_scores,
    ) in enumerate(
        zip(
            box_gt_bounds,
            box_gt_class,
            box_pred_bounds,
            box_pred_class,
            box_pred_scores,
        )
    ):
        if match_within_class:
            _classes = _gt_class.unique()
            assert len(_classes) > 0, f"No GT detections for element {i}"
            for _cls in _classes:
                # Partition the bboxes by classes and evaluate classes separately
                gt_within_cls = _gt_class == _cls
                pred_within_cls = _pred_class == _cls
                assignment_dict = get_det_assigments_to_gt(
                    box_gt_bounds=_gt_bounds[gt_within_cls],
                    box_gt_class=_gt_class[gt_within_cls],
                    box_pred_bounds=_pred_bounds[pred_within_cls],
                    box_pred_class=_pred_class[pred_within_cls],
                    box_pred_scores=_pred_scores[pred_within_cls],
                    iou_thr=iou_thr,
                    eps=eps,
                )
                cls_to_retvals[int(_cls.cpu())].append(assignment_dict)

        else:
            assignment_dict = get_det_assigments_to_gt(
                box_gt_bounds=_gt_bounds,
                box_gt_class=_gt_class,
                box_pred_bounds=_pred_bounds,
                box_pred_class=_pred_class,
                box_pred_scores=_pred_scores,
                iou_thr=iou_thr,
                eps=eps,
            )
            cls_to_retvals[all_class].append(assignment_dict)

    ap, recall = {}, {}
    for cls_id, retvals in cls_to_retvals.items():
        ap[cls_id], recall[cls_id], n_dets = compute_ap_recall(
            [r["is_true_pos"] for r in retvals],
            [r["box_pred_scores"] for r in retvals],
            n_pos=[r["gt_to_det"].shape[-1] for r in retvals],
            ap_mode=ap_mode,
        )
        if n_dets == 0:
            warnings.warn(
                RuntimeWarning(f"No detections across all scenes (class {cls_id})")
            )

    return ap, recall


def eval_bboxes_and_print(
    box_gt_bounds: Sequence[Tensor],
    box_gt_class: Sequence[Tensor],
    box_pred_bounds: Sequence[Tensor],
    box_pred_class: Sequence[Tensor],
    box_pred_scores: Sequence[Tensor],
    match_within_class: bool = True,
    iou_thr: Sequence[float] = (0.25, 0.5, 0.75),
    ap_mode: str = "area",
    label_to_cat: Dict[int, str] = None,
    eps: float = 1e-6,
):
    ALL_CLASS = 0
    if not match_within_class:
        label_to_cat = {ALL_CLASS: "all"}

    if label_to_cat is None:
        all_classes = [int(i) for i in torch.cat(box_gt_class).unique()]
        label_to_cat = dict(zip(all_classes, all_classes))

    ap, rec = compute_box_det_ap_recall(
        box_gt_bounds=box_gt_bounds,
        box_gt_class=box_gt_class,
        box_pred_bounds=box_pred_bounds,
        box_pred_class=box_pred_class,
        box_pred_scores=box_pred_scores,
        match_within_class=match_within_class,
        iou_thr=iou_thr,
        ap_mode=ap_mode,
        all_class=ALL_CLASS,
        eps=eps,
    )

    ret_dict = dict()
    header = ["classes"]
    table_columns = [[label_to_cat[label] for label in ap.keys()] + ["Overall"]]

    for i, iou_thresh in enumerate(iou_thr):
        header.append(f"AP_{iou_thresh:.2f}")
        header.append(f"AR_{iou_thresh:.2f}")
        ap_list = []
        for label in ap.keys():
            ret_dict[f"{label_to_cat[label]}_AP_{iou_thresh:.2f}"] = float(ap[label][i])
            ap_list.append(ap[label][i])
        ret_dict[f"mAP_{iou_thresh:.2f}"] = float(np.mean(ap_list))

        table_columns.append(list(map(float, ap_list)))
        table_columns[-1] += [ret_dict[f"mAP_{iou_thresh:.2f}"]]
        table_columns[-1] = [f"{x:.4f}" for x in table_columns[-1]]

        rec_list = []
        for label in rec.keys():
            ret_dict[f"{label_to_cat[label]}_rec_{iou_thresh:.2f}"] = float(
                rec[label][i]
            )
            rec_list.append(rec[label][i])
        ret_dict[f"mAR_{iou_thresh:.2f}"] = float(np.mean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f"mAR_{iou_thresh:.2f}"]]
        table_columns[-1] = [f"{x:.4f}" for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print("\n" + table.table)

    return ret_dict
