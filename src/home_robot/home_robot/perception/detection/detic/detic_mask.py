# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer
from omegaconf import DictConfig

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "Detic/third_party/CenterNet2/")
)

from centernet.config import add_centernet_config  # noqa:E402

from home_robot.perception.detection.detic.Detic.detic.config import (  # noqa:E402
    add_detic_config,
)
from home_robot.perception.detection.detic.Detic.detic.modeling.text.text_encoder import (  # noqa:E402
    build_text_encoder,
)
from home_robot.perception.detection.detic.Detic.detic.modeling.utils import (  # noqa:E402
    reset_cls_test,
)

BUILDIN_CLASSIFIER = {
    "lvis": "Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": "Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": "Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": "Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}


class Detic:
    def __init__(self, config: DictConfig, visualize: bool = False) -> None:
        if config.vocabulary == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = config.custom_vocabulary.split(",")
            classifier = self._get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[config.vocabulary]
            )
            classifier = BUILDIN_CLASSIFIER[config.vocabulary]

        self.augment_mask_with_box = config.augment_mask_with_box
        self.visualize = visualize
        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")

        self.predictor = DefaultPredictor(self.setup_cfg(config))
        reset_cls_test(self.predictor.model, classifier, num_classes)

    @staticmethod
    def _get_clip_embeddings(vocabulary: List[str], prompt: str = "a ") -> torch.Tensor:
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    @staticmethod
    def setup_cfg(config: DictConfig) -> CfgNode:
        config_file = str(Path(__file__).resolve().parent / config.config_file)
        weights = str(Path(__file__).resolve().parent / config.weights)

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            config.confidence_threshold
        )
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
            Path(__file__).resolve().parent
            / "Detic"
            / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        )
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        return cfg

    def run_on_image(self, image: np.ndarray) -> Tuple[Dict, Optional[VisImage]]:
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image)

        if not self.visualize:
            return predictions, None

        vis_output = None

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    @staticmethod
    def get_default_goal_mask(h: int, w: int) -> np.ndarray:
        m = np.ones((h, w), dtype=bool)
        m[: int(h / 3)] = False
        m[int(h - h / 8) :] = False
        m[:, : int(w / 3)] = False
        m[:, int(w - w / 3) :] = False
        return m

    @staticmethod
    def augment_mask(best_mask: np.ndarray, best_box: np.ndarray) -> np.ndarray:
        """make bottom 1/4 of box True"""
        x1, y1, x2, y2 = best_box
        y1 = y2 - ((y2 - y1) / 4)

        y2 = min(y2, best_mask.shape[0] - 1)
        best_mask[int(y1) : int(y2), int(x1) : int(x2)] = True
        return best_mask

    def get_goal_mask(
        self, goal_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[VisImage]]:
        """
        Take the highest confidence instance segmentation mask whose bounding
        box contains the image center point. If no bounding box contains the
        center point, default to a central crop.
        """
        h, w = goal_img.shape[:2]
        ch, cw = int(h / 2), int(w / 2)

        goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR)
        predictions, viz = self.run_on_image(goal_img)
        if "instances" not in predictions:
            return self.get_default_goal_mask(h, w), viz

        instances = predictions["instances"].to(self.cpu_device)
        n_instances = instances.pred_masks.shape[0]
        if n_instances == 0:
            return self.get_default_goal_mask(h, w), viz

        boxes = instances.pred_boxes.tensor.numpy().tolist()
        masks = instances.pred_masks
        scores = instances.scores

        best_mask = None
        best_box = None
        best_conf = 0.0
        for i in range(n_instances):
            conf = scores[i].item()

            x1, y1, x2, y2 = boxes[i]
            in_bbox = ch > y1 and ch < y2 and cw > x1 and cw < x2
            if in_bbox and conf > best_conf:
                best_mask = masks[i]
                best_box = boxes[i]
                best_conf = conf

        if best_mask is None:
            return self.get_default_goal_mask(h, w), viz

        if self.augment_mask_with_box:
            best_mask = self.augment_mask(best_mask, best_box)
        return best_mask.numpy(), viz
