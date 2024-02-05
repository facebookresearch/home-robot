import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "Grounded-Segment-Anything/EfficientSAM")
)
from MobileSAM.setup_mobile_sam import setup_model  # noqa: E402
from segment_anything import SamAutomaticMaskGenerator, SamPredictor  # noqa: E402

from home_robot.core.abstract_perception import PerceptionModule  # noqa: E402
from home_robot.core.interfaces import Observations  # noqa: E402
from home_robot.perception.detection.utils import (  # noqa: E402
    filter_depth,
    overlay_masks,
)
from home_robot.utils.data_tools.dict import update as dict_update  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = Path(__file__).resolve().parent
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = str(
    PARENT_DIR
    / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = str(
    PARENT_DIR / "checkpoints" / "groundingdino_swint_ogc.pth"
)
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "checkpoints" / "mobile_sam.pt")
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

_DEFAULT_MASK_GENERATOR_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class SAMPerception(PerceptionModule):
    def __init__(
        self,
        custom_vocabulary: List[str] = "['', 'dog', 'grass', 'sky']",
        gpu_device_id=None,
        checkpoint_file: str = MOBILE_SAM_CHECKPOINT_PATH,
        verbose=False,
        text_threshold: float = None,
        mask_generator_kwargs: Dict[str, Any] = _DEFAULT_MASK_GENERATOR_KWARGS,
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            checkpoint_file: path to model checkpoint
            verbose: whether to print out debug information
        """

        # Building MobileSAM predictor
        checkpoint = torch.load(checkpoint_file)
        self.mobile_sam = setup_model()
        self.mobile_sam.load_state_dict(checkpoint, strict=True)
        device = gpu_device_id
        if device is None:
            device = DEVICE
        self.mobile_sam.to(device=device)
        self.custom_vocabulary = custom_vocabulary
        self.sam_predictor = SamPredictor(self.mobile_sam)
        final_mask_generator_kwargs = copy.deepcopy(_DEFAULT_MASK_GENERATOR_KWARGS)
        final_mask_generator_kwargs = dict_update(
            final_mask_generator_kwargs, mask_generator_kwargs
        )
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.mobile_sam, **final_mask_generator_kwargs
        )

    def reset_vocab(self, new_vocab: List[str]):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        self.custom_vocabulary = new_vocab

    def generate(self, image, min_area=1600):
        masks = self.mask_generator.generate(image)
        returned_masks = []
        for mask in masks:
            # filter out masks that are too small
            if mask["area"] >= min_area:
                returned_masks.append(mask["segmentation"])
        return returned_masks

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = False,
    ) -> np.ndarray:
        """
        Get masks using SAM
        Arguments:
            image: image of shape (H, W, 3)
        Returns:
            masks: masks of shape (N, H, W)
        """
        height, width, _ = obs.rgb.shape
        image = obs.rgb
        if not image.dtype == np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)

        masks = np.array(self.generate(image))

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.savefig("original.png", dpi=100)
        show_anns(masks)
        plt.axis("off")
        plt.savefig("segmented.png", dpi=100)

        if depth_threshold is not None and obs.depth is not None:
            masks = np.array(
                [filter_depth(mask, obs.depth, depth_threshold) for mask in masks]
            )
        semantic_map, instance_map = overlay_masks(
            masks, np.zeros(masks.shape[0]), (height, width)
        )

        obs.semantic = semantic_map.astype(int)
        obs.instance = instance_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map

        # random filling object classes -- right now using cups
        obs.task_observations["instance_classes"] = np.full(masks.shape[0], 31)
        obs.task_observations["instance_scores"] = np.ones(masks.shape[0])
        obs.task_observations["semantic_frame"] = None
        return obs
