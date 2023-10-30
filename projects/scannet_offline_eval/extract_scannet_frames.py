# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import dataclasses
import random
import sys
import timeit
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from home_robot.datasets.scannet import ScanNetDataset, ScanNetModalities
from home_robot.datasets.scannet.scannet_constants import SCANNET_COLOR_MAP_300

# from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

colors = list(SCANNET_COLOR_MAP_300.values())
random.shuffle(colors)


# Import necessary libraries
import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.visualizer import (
    ColorMode,
    GenericMask,
    Visualizer,
    _create_text_labels,
)


def show_instance_image(
    rgb,
    instance_map,
    class_list,
    class_id_to_name,
    scale=1.0,
    custom_class_colors=None,
    custom_draw_fn=None,
):
    # original_image: numpy array with shape (H, W, 3)
    # instance_image: torch.long tensor with shape (H, W)
    # class_list: list of ints representing classes

    assert rgb.max() <= 1.0
    original_image = (rgb.cpu().numpy() * 255).astype(
        np.uint8
    )  # replace with your image path or method to load image
    instance_image_np = (
        instance_map.cpu().numpy()
    )  # converting torch tensor to numpy array

    # Extracting unique instances and their counts
    instances_ids = np.unique(instance_image_np)

    # Creating masks and bounding boxes for each instance
    masks = [instance_image_np == i for i in instances_ids if i != -1]
    boxes = [cv2.boundingRect(np.uint8(mask)) for mask in masks]  # returns (x,y,w,h)

    # Convert (x,y,w,h) to (x1, y1, x2, y2) format
    boxes = [(x, y, x + w, y + h) for x, y, w, h in boxes]

    # Creating a dict for detected instances
    height, width = original_image.shape[:2]
    instances_obj = Instances((height, width))
    instances_obj.pred_boxes = Boxes(boxes)
    instances_obj.pred_masks = masks
    if class_list is not None:
        instances_obj.pred_classes = torch.as_tensor(class_list, dtype=torch.int64)
        # Prepare metadata for visualization
        # Assuming COCO dataset categories for example. Replace with your own if different.
    else:
        instances_obj.pred_classes = torch.as_tensor(instances_ids, dtype=torch.int64)

    if class_id_to_name is not None:
        class_names = list(class_id_to_name.values())
    else:
        class_names = {i: str(i) for i in instances_ids}

    # Clear the existing metadata if it's safe to do so
    dataset_name = "segmenter_data"
    if dataset_name in MetadataCatalog:
        MetadataCatalog.remove(dataset_name)
    # Create (or get) a MetadataCatalog for your dataset and set its thing_classes attribute
    metadata = MetadataCatalog.get(dataset_name)
    metadata.set(thing_classes=class_names)
    if custom_class_colors is not None:
        metadata.thing_colors = custom_class_colors

    # Visualize the results
    v = Visualizer(original_image[:, :, ::-1], metadata=metadata, scale=scale)
    if custom_draw_fn is None:
        out = v.draw_instance_predictions(instances_obj)
    else:
        out = custom_draw_fn(v, instances_obj)
    visualized_image = out.get_image()[:, :, ::-1]  # convert from RGB to BGR
    return visualized_image


def draw_instance_predictions(self, predictions):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    Returns:
        output (VisImage): image object with visualizations.
    """
    self._instance_mode = ColorMode.SEGMENTATION
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = (
        predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    )
    labels = _create_text_labels(
        classes, scores, self.metadata.get("thing_classes", None)
    )
    keypoints = (
        predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    )

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
    else:
        masks = None

    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
        "thing_colors"
    ):
        # colors = [
        #     self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
        # ]
        colors = [[x / 255 for x in self.metadata.thing_colors[c]] for c in classes]
        alpha = 0.7
    else:
        colors = None
        alpha = 0.5

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(
            self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
        )
        alpha = 0.3

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def add_annotated_frames(output_folder, data, colors, new_output_folder):
    out_idx, _, scene_name = output_folder.name.split("-")
    ds_idx = data.scene_list.index(scene_name)  #'scene0000_00'
    print(f"---- loading {scene_name}")
    scene_obs = data.__getitem__(ds_idx, show_progress=True)
    for frame_idx, (rgb, instance2d) in enumerate(
        zip(scene_obs["images"], scene_obs["instance_2ds"])
    ):
        inst_im = show_instance_image(
            rgb,
            instance2d,
            None,
            {i: str(i) for i in range(0, 256)},
            scale=0.7,
            custom_class_colors=colors[:256],
            custom_draw_fn=draw_instance_predictions,
        )
        img = Image.fromarray(inst_im.astype("uint8"))
        if new_output_folder is None:
            new_output_folder = output_folder
        img.save(new_output_folder / f"{frame_idx:06d}-annotated.png")
    print(f"---- finished {scene_name}")


if __name__ == "__main__":
    val_data = ScanNetDataset(
        root_dir="/private/home/ssax/home-robot/src/home_robot/home_robot/datasets/scannet/data",
        frame_skip=1,
        n_classes=50,
        split="val",
        load_only_first_k_frames=600,
        skipnan=False,
    )

    p = Path("/checkpoint/maksymets/eaif/datasets/eqa-v2/frames/scannet-v0/")
    paths = list(p.iterdir())
    for scene_folder in tqdm(paths):
        out_idx, _, scene_name = scene_folder.name.split("-")
        if scene_name not in val_data.scene_list:
            continue
        add_annotated_frames(
            scene_folder,
            val_data,
            colors,
            None,
        )
