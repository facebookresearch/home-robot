import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
from segment_anything import SamPredictor  # noqa: E402

from home_robot.core.abstract_perception import PerceptionModule  # noqa: E402
from home_robot.core.interfaces import Observations  # noqa: E402
from home_robot.perception.detection.utils import (  # noqa: E402
    filter_depth,
    overlay_masks,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = str(
    Path(__file__).resolve().parent
    / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
MOBILE_SAM_CHECKPOINT_PATH = "./mobile_sam.pt"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


class GroundedSAMPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        custom_vocabulary: List[str] = "['', 'dog', 'grass', 'sky']",
        sem_gpu_id=None,
        checkpoint_file: str = MOBILE_SAM_CHECKPOINT_PATH,
        verbose=False,
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
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )

        # Building MobileSAM predictor
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        self.mobile_sam = setup_model()
        self.mobile_sam.load_state_dict(checkpoint, strict=True)
        self.mobile_sam.to(device=DEVICE)
        self.custom_vocabulary = custom_vocabulary
        self.sam_predictor = SamPredictor(self.mobile_sam)

    def reset_vocab(self, new_vocab: List[str]):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        self.custom_vocabulary = new_vocab

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional = None,
        draw_instance_predictions: bool = False,
    ):
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """

        if draw_instance_predictions:
            raise NotImplementedError
        # Predict classes and hyper-param for GroundingDINO
        CLASSES = self.custom_vocabulary

        height, width, _ = obs.rgb.shape
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=obs.rgb,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD,
        )
        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = self.segment(image=obs.rgb, xyxy=detections.xyxy)

        if depth_threshold is not None and obs.depth is not None:
            detections.mask = np.array(
                [
                    filter_depth(mask, obs.depth, depth_threshold)
                    for mask in detections.mask
                ]
            )

        semantic_map, instance_map = overlay_masks(
            detections.mask, detections.class_id, (height, width)
        )

        obs.semantic = semantic_map.astype(int)
        if obs.task_observations is None:
            obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = detections.class_id
        obs.task_observations["instance_scores"] = detections.confidence
        obs.task_observations["semantic_frame"] = None
        return obs

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
