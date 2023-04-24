#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def rad2deg(tensor):
    return 180.0 * tensor / math.pi


# ============================= Pose estimation metrics ===============================


def compute_translation_error(pred_pose, gt_pose, reduction="mean"):
    """
    Computes the error (meters) in translation components of pose prediction.
    Inputs:
        pred_pose - (bs, 3) --- (x, y, theta)
        gt_pose   - (bs, 3) --- (x, y, theta)

    Note: x, y must be in meters.
    """
    error = torch.sqrt(
        F.mse_loss(pred_pose[:, :2], gt_pose[:, :2], reduction=reduction)
    )
    return error


def compute_angular_error(pred_pose, gt_pose, reduction="mean"):
    """
    Computes the error (degrees) in rotation components of pose prediction.
    Inputs:
        pred_pose - (bs, 3) --- (x, y, theta)
        gt_pose   - (bs, 3) --- (x, y, theta)

    Note: theta must be in radians.
    """
    angular_diff = pred_pose[:, 2] - gt_pose[:, 2]
    normalized_angular_diff = torch.atan2(
        torch.sin(angular_diff), torch.cos(angular_diff)
    )
    error = torch.abs(normalized_angular_diff)

    if reduction == "mean":
        error = error.mean()
    elif reduction == "none":
        error = error
    elif reduction == "sum":
        error = error.sum()

    error = rad2deg(error)

    return error


def measure_pose_estimation_performance(pred_pose, gt_pose, reduction="mean"):
    trans_error = compute_translation_error(pred_pose, gt_pose, reduction=reduction)
    ang_error = compute_angular_error(pred_pose, gt_pose, reduction=reduction)
    if reduction != "none":
        trans_error = trans_error.item()
        ang_error = ang_error.item()

    metrics = {"translation_error": trans_error, "angular_error": ang_error}

    return metrics


# ============================= Area coverage metrics =================================


def measure_area_seen_performance(map_states, map_scale=1.0, reduction="mean"):
    """
    Inputs:
        map_states - (bs, 2, M, M) world map with channel 0 representing occupied
                     regions (1s) and channel 1 representing explored regions (1s)
    """

    bs = map_states.shape[0]
    explored_map = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
    occ_space_map = (map_states[:, 0] > 0.5).float() * explored_map  # (bs, M, M)
    free_space_map = (map_states[:, 0] <= 0.5).float() * explored_map  # (bs, M, M)

    all_cells_seen = explored_map.view(bs, -1).sum(dim=1)  # (bs, )
    occ_cells_seen = occ_space_map.view(bs, -1).sum(dim=1)  # (bs, )
    free_cells_seen = free_space_map.view(bs, -1).sum(dim=1)  # (bs, )

    area_seen = all_cells_seen * (map_scale**2)
    free_space_seen = free_cells_seen * (map_scale**2)
    occupied_space_seen = occ_cells_seen * (map_scale**2)

    if reduction == "mean":
        area_seen = area_seen.mean().item()
        free_space_seen = free_space_seen.mean().item()
        occupied_space_seen = occupied_space_seen.mean().item()
    elif reduction == "sum":
        area_seen = area_seen.sum().item()
        free_space_seen = free_space_seen.sum().item()
        occupied_space_seen = occupied_space_seen.sum().item()

    return {
        "area_seen": area_seen,
        "free_space_seen": free_space_seen,
        "occupied_space_seen": occupied_space_seen,
    }


def reduce_metrics(metrics, reduction):
    if type(metrics) == type({}):
        r_metrics = {}
        if reduction == "mean":
            for k, v in metrics.items():
                r_metrics[k] = v.mean()
        elif reduction == "sum":
            for k, v in metrics.items():
                r_metrics[k] = v.sum()
        else:
            for k, v in metrics.items():
                r_metrics[k] = v
    else:
        if reduction == "mean":
            r_metrics = metrics.mean()
        elif reduction == "sum":
            r_metrics = metrics.sum()
        else:
            r_metrics = metrics

    return r_metrics


# =============================== Map quality metrics =================================


def process_predictions(preds, entropy_thresh=0.35):
    """
    Inputs:
        preds - (N, 2, H, W) Tensor values between 0.0 to 1.0
              - channel 0 predicts probability of occupied space
              - channel 1 predicts probability of explored space
        entropy_thresh - predictions with entropy larger than this value are discarded
    """
    N, _, H, W = preds.shape
    preds = preds.clone()
    preds = preds.permute(0, 2, 3, 1)
    preds = preds.contiguous()  # (N, H, W, C)

    # Compute entropy
    probs = preds[..., 1]
    log_probs = (probs + 1e-12).log()
    log_1_probs = (1 - probs + 1e-12).log()
    entropy = -probs * log_probs - (1 - probs) * log_1_probs  # (N, H, W)

    max_entropy = math.log(2.0)
    entropy_np = (entropy / max_entropy).cpu().numpy()
    entropy_image = entropy_np * 255.0
    entropy_image = np.stack(
        [entropy_image, entropy_image, entropy_image], axis=3
    )  # (N, H, W, C)
    entropy_image = entropy_image.astype(np.uint8)

    preds = preds.cpu().numpy()  # (N, H, W, 2)
    exp_mask = (preds[..., 1] > 0.5).astype(np.float32)
    occ_mask = (preds[..., 0] > 0.5).astype(np.float32) * exp_mask
    free_mask = (preds[..., 0] <= 0.5).astype(np.float32) * exp_mask
    unk_mask = 1 - exp_mask

    # Occupied regions are blue, free regions are green.
    # Modulate the values based on confidence
    pred_imgs = np.stack(
        [
            0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
            255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
        ],
        axis=3,
    ).astype(
        np.uint8
    )  # (N, H, W, 3)

    # Occupied regions are blue, free regions are green.
    # Filter out the uncertain predictions
    entropy_mask = (entropy_np <= entropy_thresh).astype(np.float32)
    free_mask_ = free_mask * entropy_mask
    occ_mask_ = occ_mask * entropy_mask
    unk_mask_ = np.clip(unk_mask + (1 - entropy_mask), 0, 1)

    pred_imgs_filtered = np.stack(
        [
            0.0 * occ_mask_ + 0.0 * free_mask_ + 255.0 * unk_mask_,
            0.0 * occ_mask_ + 255.0 * free_mask_ + 255.0 * unk_mask_,
            255.0 * occ_mask_ + 0.0 * free_mask_ + 255.0 * unk_mask_,
        ],
        axis=3,
    ).astype(
        np.uint8
    )  # (N, H, W, 3)

    return pred_imgs, pred_imgs_filtered, entropy_image


def reduce_metrics(metrics, reduction):
    if type(metrics) == type({}):
        r_metrics = {}
        if reduction == "mean":
            for k, v in metrics.items():
                r_metrics[k] = v.mean()
        elif reduction == "sum":
            for k, v in metrics.items():
                r_metrics[k] = v.sum()
        else:
            for k, v in metrics.items():
                r_metrics[k] = v
    else:
        if reduction == "mean":
            r_metrics = metrics.mean()
        elif reduction == "sum":
            r_metrics = metrics.sum()
        else:
            r_metrics = metrics

    return r_metrics


def batched_occ_metrics(
    pred_occupancy, gt_occupancy, reduction="mean", apply_mask=False
):
    """
    Measures the precision, recall of free space, occupancy and overall accuracy of predictions.
    Ignores the predictions in unknown parts if apply_mask is set.

    Inputs:
        pred_occupancy - (bs, H, W, C)
        gt_occupancy - (bs, H, W, C)
    """
    # Preprocess data
    pred_free_space = np.all(
        pred_occupancy == np.array([0, 255, 0]), axis=-1
    )  # (bs, H, W)
    pred_occ_space = np.all(
        pred_occupancy == np.array([0, 0, 255]), axis=-1
    )  # (bs, H, W)
    gt_free_space = np.all(gt_occupancy == np.array([0, 255, 0]), axis=-1)  # (bs, H, W)
    gt_occ_space = np.all(gt_occupancy == np.array([0, 0, 255]), axis=-1)  # (bs, H, W)
    valid_mask = gt_free_space | gt_occ_space  # (bs, H, W)

    # Mask out predictions by the valid areas of GT
    if apply_mask:
        pred_free_space = pred_free_space & valid_mask
        pred_occ_space = pred_occ_space & valid_mask

    # Accuracy metrics
    total_gt_free_space = gt_free_space.sum((1, 2)).astype(np.float32)  # (bs, )
    total_pred_free_space = pred_free_space.sum((1, 2)).astype(np.float32)  # (bs, )
    tp_free_space = (
        (pred_free_space & gt_free_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    free_space_recall = tp_free_space / (total_gt_free_space + EPS)  # (bs, )
    free_space_prec = tp_free_space / (total_pred_free_space + EPS)  # (bs, )
    free_space_f1 = (
        2
        * free_space_prec
        * free_space_recall
        / (free_space_prec + free_space_recall + EPS)
    )

    total_gt_occ_space = gt_occ_space.sum((1, 2)).astype(np.float32)  # (bs, )
    total_pred_occ_space = pred_occ_space.sum((1, 2)).astype(np.float32)  # (bs, )
    tp_occ_space = (
        (pred_occ_space & gt_occ_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    occ_space_recall = tp_occ_space / (total_gt_occ_space + EPS)  # (bs, )
    occ_space_prec = tp_occ_space / (total_pred_occ_space + EPS)  # (bs, )
    occ_space_f1 = (
        2
        * occ_space_prec
        * occ_space_recall
        / (occ_space_prec + occ_space_recall + EPS)
    )  # (bs, )

    overall_acc = (tp_free_space + tp_occ_space) / (
        total_gt_free_space + total_gt_occ_space + EPS
    )  # (bs, )

    # IoU metrics
    free_space_intersection = (
        (pred_free_space & gt_free_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    free_space_union = (
        (pred_free_space | gt_free_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    free_space_iou = free_space_intersection / (free_space_union + EPS)  # (bs, )

    occ_space_intersection = (
        (pred_occ_space & gt_occ_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    occ_space_union = (
        (pred_occ_space | gt_occ_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    occ_space_iou = occ_space_intersection / (occ_space_union + EPS)  # (bs, )

    mean_iou = (free_space_iou + occ_space_iou) / 2.0
    mean_f1 = (free_space_f1 + occ_space_f1) / 2.0

    metrics = {
        "overall_acc": overall_acc,
        "free_space_recall": free_space_recall,
        "free_space_prec": free_space_prec,
        "free_space_f1": free_space_f1,
        "free_space_iou": free_space_iou,
        "occ_space_recall": occ_space_recall,
        "occ_space_prec": occ_space_prec,
        "occ_space_f1": occ_space_f1,
        "occ_space_iou": occ_space_iou,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
    }

    return reduce_metrics(metrics, reduction)


def batched_anticipative_metrics(
    pred_occupancy, gt_occupancy, reduction="mean", apply_mask=False
):
    """
    Measures the intersection of free-space, occupied-space and overall space
    in pred_occupancy and gt_occupancy

    pred_occupancy - (bs, h, w, 2) numpy arrays
    gt_occupancy - (bs, h, w, 2) numpy arrays
    """
    pred_free_space = np.all(
        pred_occupancy == np.array([0, 255, 0]), axis=-1
    )  # (bs, H, W)
    pred_occ_space = np.all(
        pred_occupancy == np.array([0, 0, 255]), axis=-1
    )  # (bs, H, W)
    gt_free_space = np.all(gt_occupancy == np.array([0, 255, 0]), axis=-1)  # (bs, H, W)
    gt_occ_space = np.all(gt_occupancy == np.array([0, 0, 255]), axis=-1)  # (bs, H, W)
    valid_mask = gt_free_space | gt_occ_space  # (bs, H, W)

    # Mask out predictions by the valid areas of GT
    if apply_mask:
        pred_free_space = pred_free_space & valid_mask
        pred_occ_space = pred_occ_space & valid_mask

    free_space_covered = (
        (pred_free_space & gt_free_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    occ_space_covered = (
        (pred_occ_space & gt_occ_space).sum((1, 2)).astype(np.float32)
    )  # (bs, )
    map_accuracy = free_space_covered + occ_space_covered

    metrics = {"map_accuracy": map_accuracy}

    return reduce_metrics(metrics, reduction)


def measure_map_quality(
    pred_maps,
    gt_maps,
    map_scale,
    entropy_thresh=0.35,
    reduction="mean",
    apply_mask=False,
):
    """
    Inputs:
        pred_maps - (bs, 2, H, W) Tensor maps
        gt_maps   - (bs, 2, H, W) Tensor maps

    Channel 0 - probability of occupied space
    Channel 1 - probability of explored space
    """
    device = pred_maps.device
    proc_pred_maps = process_predictions(pred_maps, entropy_thresh)[1]  # (bs, H, W, 3)
    proc_gt_maps = process_predictions(gt_maps)[1]  # (bs, H, W, 3)

    curr_occ_metrics = batched_occ_metrics(
        proc_pred_maps,
        proc_gt_maps,
        reduction=reduction,
        apply_mask=apply_mask,
    )
    curr_anticipative_metrics = batched_anticipative_metrics(
        proc_pred_maps,
        proc_gt_maps,
        reduction=reduction,
        apply_mask=apply_mask,
    )
    mean_iou = (
        curr_occ_metrics["free_space_iou"] + curr_occ_metrics["occ_space_iou"]
    ) / 2.0

    all_metrics = {
        "mean_iou": mean_iou,
        "free_space_iou": curr_occ_metrics["free_space_iou"],
        "occ_space_iou": curr_occ_metrics["occ_space_iou"],
        "map_accuracy": curr_anticipative_metrics["map_accuracy"] * (map_scale**2),
    }

    if reduction == "mean" or reduction == "sum":
        metrics = {k: v.item() for k, v in all_metrics.items()}
    else:
        metrics = all_metrics

    return metrics


def measure_anticipation_reward(
    pred_maps,
    gt_maps,
    reduction="mean",
    apply_mask=False,
):
    """
    Inputs:
        pred_maps - (bs, 2, H, W) Tensor maps
        gt_maps   - (bs, 2, H, W) Tensor maps

    Channel 0 - probability of occupied space
    Channel 1 - probability of explored space
    """
    pred_explored_space = pred_maps[:, 1] > 0.5  # (bs, H, W)
    pred_free_space = (pred_maps[:, 0] <= 0.5) & pred_explored_space
    pred_occ_space = (pred_maps[:, 0] > 0.5) & pred_explored_space

    gt_explored_space = gt_maps[:, 1] > 0.5
    gt_free_space = (gt_maps[:, 0] <= 0.5) & gt_explored_space
    gt_occ_space = (gt_maps[:, 0] > 0.5) & gt_explored_space

    if apply_mask:
        pred_free_space = pred_free_space & gt_explored_space
        pred_occ_space = pred_occ_space & gt_explored_space

    free_space_covered = (pred_free_space & gt_free_space).sum(dim=2).sum(dim=1)
    occ_space_covered = (pred_occ_space & gt_occ_space).sum(dim=2).sum(dim=1)
    area_covered = (free_space_covered + occ_space_covered).float()

    return reduce_metrics(area_covered, reduction)


class Metric:
    def __init__(self):
        self.reset()

    def update(self, val, size):
        self._metric += val
        self._count += size
        self._metric_list.append(val)
        self._count_list.append(size)

    def get_metric(self):
        return self._metric / (self._count + EPS)

    @property
    def metric_list(self):
        return copy.deepcopy(self._metric_list)

    @property
    def count_list(self):
        return copy.deepcopy(self._count_list)

    def reset(self):
        self._metric = 0.0
        self._count = 0.0
        self._metric_list = []
        self._count_list = []


class TemporalMetric:
    def __init__(self):
        self.reset()

    def update(self, val, size, time):
        if time not in self._metric:
            self._metric[time] = Metric()
            self._count[time] = Metric()
        self._metric[time].update(val, size)

    def get_metric(self):
        metrics = {}
        for time in self._metric.keys():
            metrics[time] = self._metric[time].get_metric()
        return metrics

    def get_last_metric(self):
        times = sorted(list(self._metric.keys()))
        return self._metric[times[-1]].get_metric()

    @property
    def metric_list(self):
        metrics = {}
        for time in self._metric.keys():
            metrics[time] = self._metric[time].metric_list
        return metrics

    @property
    def count_list(self):
        counts = {}
        for time in self._metric.keys():
            counts[time] = self._metric[time].count_list
        return counts

    def reset(self):
        self._metric = {}
        self._count = {}
