# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer

from home_robot.core.interfaces import Observations

from .coco_categories import coco_categories, coco_categories_mapping


class COCOMaskRCNN:
    def __init__(
        self,
        vocabulary: str = "coco",
        sem_pred_prob_thr: float = 0.5,
        sem_gpu_id: int = 0,
    ):
        """
        Arguments:
            vocabulary: currently one of "coco" for indoor coco categories or "coco-subset"
             for 6 coco goal categories
            sem_pred_prob_thr: prediction threshold
            sem_gpu_id: prediction GPU id (-1 for CPU)
        """
        self.segmentation_model = ImageSegmentation(sem_pred_prob_thr, sem_gpu_id)
        self.visualize = True
        if vocabulary == "coco":
            self.vocabulary = coco_categories
            self.vocabulary_mapping = coco_categories_mapping
            self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        elif vocabulary == "coco-subset":
            self.vocabulary = {
                "chair": 0,
                "couch": 1,
                "plant": 2,
                "bed": 3,
                "toilet": 4,
                "tv": 5,
            }
            self.vocabulary_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
            }
            self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        else:
            raise ValueError("Vocabulary {} does not exist".format(vocabulary))
        self.num_sem_categories = len(self.vocabulary)

    def get_prediction(
        self, images: np.ndarray, depths: Optional[np.ndarray] = None, conf_thresh=0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arguments:
            images: images of shape (batch_size, H, W, 3) (in BGR order)
            depths: depth frames of shape (batch_size, H, W)

        Returns:
            one_hot_predictions: one hot segmentation predictions of shape
             (batch_size, H, W, num_sem_categories)
            visualizations: prediction visualization images
             shape (batch_size, H, W, 3) if self.visualize=True, else
             original images
        """
        batch_size, height, width, _ = images.shape

        predictions, visualizations = self.segmentation_model.get_predictions(
            images, visualize=self.visualize
        )
        one_hot_predictions = np.zeros(
            (batch_size, height, width, self.num_sem_categories + 1)
        )

        for i in range(batch_size):
            for j, class_idx in enumerate(
                predictions[i]["instances"].pred_classes.cpu().numpy()
            ):
                if class_idx in list(self.vocabulary_mapping.keys()):
                    idx = self.vocabulary_mapping[class_idx]
                    obj_mask = predictions[i]["instances"].pred_masks[j] * 1.0
                    obj_mask = obj_mask.cpu().numpy()
                    score = predictions[i]["instances"].scores[j].item()
                    if score < conf_thresh:
                        continue
                    if depths is not None:
                        depth = depths[i]
                        md = np.median(depth[obj_mask == 1])
                        if md == 0:
                            filter_mask = np.ones_like(obj_mask, dtype=bool)
                        else:
                            # Restrict objects to 1m depth
                            filter_mask = (depth >= md + 50) | (depth <= md - 50)
                        obj_mask[filter_mask] = 0.0

                    one_hot_predictions[i, :, :, idx] += obj_mask

        if self.visualize:
            visualizations = np.stack([vis.get_image() for vis in visualizations])
        else:
            # Convert BGR to RGB for visualization
            visualizations = images[:, :, :, ::-1]

        return one_hot_predictions, visualizations

    def predict(
        self,
        obs: Observations,
        confidence_threshold: Optional[float] = None,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in BGR order)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W, N) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        images = obs.rgb[np.newaxis, :, :, :]
        depths = obs.depth[np.newaxis, :, :]
        pred, vis = self.get_prediction(images, depths, confidence_threshold)
        pred, vis = pred[0], vis[0]
        obs.semantic = pred
        obs.task_observations["semantic_frame"] = vis
        return obs


class ImageSegmentation:
    def __init__(self, sem_pred_prob_thr, sem_gpu_id):
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
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, images, visualize=False):
        return self.demo.run_on_images(images, visualize=visualize)


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
        # t0 = time.time()

        predictions = self.predictor(images)
        batch_size = len(predictions)
        visualizations = []

        # Convert BGR to RGB for visualization
        images = images[:, :, :, ::-1]

        # t1 = time.time()
        # print(f"[Obs preprocessing] Segmentation prediction time: {t1 - t0:.2f}")

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

        # t2 = time.time()
        # print(f"[Obs preprocessing] Segmentation visualization time: {t2 - t1:.2f}")

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
