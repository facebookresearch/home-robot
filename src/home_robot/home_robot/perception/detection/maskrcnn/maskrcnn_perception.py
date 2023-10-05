# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.utils import filter_depth, overlay_masks

from .coco_categories import coco_categories, coco_categories_mapping


class MaskRCNNPerception(PerceptionModule):
    def __init__(
        self,
        sem_pred_prob_thr: float = 0.9,
        sem_gpu_id: int = 0,
    ):
        """Load MaskRCNN model trained on COCO categories for inference.

        Arguments:
            sem_pred_prob_thr: prediction threshold
            sem_gpu_id: prediction GPU id (-1 for CPU)
        """
        config_path = Path(__file__).resolve().parent / "mask_rcnn_R_50_FPN_3x.yaml"
        string_args = f"""
            --config-file {config_path}
            --input input1.jpeg
            --confidence-threshold {sem_pred_prob_thr}
            --opts MODEL.WEIGHTS
            detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
            """

        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.model = VisualizationDemo(cfg)
        self.num_sem_categories = len(coco_categories)

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - MaskRCNN expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        image = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
        depth = obs.depth
        height, width, _ = image.shape

        if obs.task_observations is None:
            obs.task_observations = {}

        predictions, visualizations = self.model.run_on_images(
            image[np.newaxis], visualize=True
        )

        pred = predictions[0]
        obs.task_observations["semantic_frame"] = visualizations[0].get_image()

        masks = pred["instances"].pred_masks.cpu().numpy()
        class_idcs = pred["instances"].pred_classes.cpu().numpy()
        scores = pred["instances"].scores.cpu().numpy()

        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )

        # Keep only relevant COCO categories
        relevant_masks = []
        relevant_class_idcs = []
        relevant_scores = []
        for i, class_idx in enumerate(class_idcs):
            if class_idx in coco_categories_mapping:
                relevant_masks.append(masks[i])
                relevant_class_idcs.append(coco_categories_mapping[class_idx] + 1)
                relevant_scores.append(scores[i])

        if len(relevant_masks) > 0:
            masks = np.stack(relevant_masks)
            class_idcs = np.stack(relevant_class_idcs)
            scores = np.stack(relevant_scores)
            semantic_map, instance_map = overlay_masks(
                masks, class_idcs, (height, width)
            )
        else:
            semantic_map = np.zeros((height, width))
            instance_map = -np.ones((height, width))

        obs.semantic = semantic_map.astype(int)
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = class_idcs
        obs.task_observations["instance_scores"] = scores

        return obs


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_images(
        self, images: np.ndarray, visualize=False
    ) -> Tuple[List[dict], List[VisImage]]:
        """
        Arguments:
            images: images of shape (batch_size, H, W, 3) (in BGR order)
            visualize: if True, return prediction visualization

        Returns:
            predictions: a list of predictions for all images
            visualizations: a list of prediction visualizations for all images
        """
        predictions = self.predictor(images)
        batch_size = len(predictions)
        visualizations = []

        # Convert BGR to RGB for visualization
        images = images[:, :, :, ::-1]

        if visualize:
            for i in range(batch_size):
                pred = predictions[i]
                image = images[i]
                visualizer = Visualizer(
                    image, self.metadata, instance_mode=self.instance_mode
                )
                if "panoptic_seg" in pred:
                    panoptic_seg, segments_info = pred["panoptic_seg"]
                    vis = visualizer.draw_panoptic_seg_predictions(
                        panoptic_seg.to(self.cpu_device), segments_info
                    )
                else:
                    if "sem_seg" in pred:
                        vis = visualizer.draw_sem_seg(
                            pred["sem_seg"].argmax(dim=0).to(self.cpu_device)
                        )
                    if "instances" in pred:
                        instances = pred["instances"].to(self.cpu_device)
                        vis = visualizer.draw_instance_predictions(
                            predictions=instances
                        )
                visualizations.append(vis)

        return predictions, visualizations


class BatchPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, images: np.ndarray) -> List[dict]:
        """
        Arguments:
            images: images of shape (batch_size, H, W, 3) (in BGR order)

        Returns:
            predictions: a list of predictions for all images
        """
        inputs = []
        for original_image in images:
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            instance = {"image": image, "height": height, "width": width}
            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
