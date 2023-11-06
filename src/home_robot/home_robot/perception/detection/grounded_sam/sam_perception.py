import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
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

    def generate(self, image):
        return self.mask_generator.generate(image)

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
