import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from MobileSAM.setup_mobile_sam import setup_model
from segment_anything import SamPredictor

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def overlay_masks(
    masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Overlays the masks of objects
    Determines the order of masks based on mask size
    """
    mask_sizes = [np.sum(mask) for mask in masks]
    sorted_mask_idcs = np.argsort(mask_sizes)

    semantic_mask = np.zeros(shape)
    instance_mask = -np.ones(shape)
    for i_mask in sorted_mask_idcs[::-1]:  # largest to smallest
        semantic_mask[masks[i_mask].astype(bool)] = class_idcs[i_mask]
        instance_mask[masks[i_mask].astype(bool)] = i_mask

    return semantic_mask, instance_mask


# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = (
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
MOBILE_SAM_CHECKPOINT_PATH = "./EfficientSAM/mobile_sam.pt"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


class GroundedMobileSAM(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        custom_vocabulary: List[str] = "['', 'dog', 'grass', 'sky']",
        checkpoint_file=MOBILE_SAM_CHECKPOINT_PATH,
    ):
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
        self.custom_vocabulary = new_vocab

    def predict(self, obs: Observations, depth_threshold: Optional = None):
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
        detections.mask = self.segment(
            image=cv2.cvtColor(obs.rgb, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy
        )
        semantic_map, instance_map = overlay_masks(
            detections.mask, detections.class_id, (height, width)
        )

        obs.semantic = semantic_map.astype(int)
        obs.task_observations = dict()
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = detections.class_id
        obs.task_observations["instance_scores"] = detections.confidence
        return obs

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def read_images_from_folder(self, folder_name: str):
        # get file in folder_name whose name starts with "rgb"
        rgb_file = [f for f in os.listdir(folder_name) if f.startswith("rgb")][0]

        object_name, start_recep = rgb_file.split(".")[0].split("-")[1:3]

        object_name = object_name.replace("_", " ")
        start_recep = start_recep.replace("_", " ")
        rgb = cv2.imread(os.path.join(folder_name, rgb_file))

        # load groundtruth object segmentation
        gt_object_mask = (
            cv2.imread(os.path.join(folder_name, "obj_gt.png"), cv2.IMREAD_GRAYSCALE)
            // 255
        )

        # load groundtruth recetacle segmentation
        gt_recep_mask = (
            cv2.imread(
                os.path.join(folder_name, "recep_seg_gt.png"), cv2.IMREAD_GRAYSCALE
            )
            // 255
        )
        return rgb, gt_object_mask, gt_recep_mask, object_name, start_recep

    def compute_iou(self, gt_object_mask, pred_mask):
        return np.maximum(
            np.sum(gt_object_mask * pred_mask)
            / np.sum(gt_object_mask + pred_mask - (gt_object_mask * pred_mask)),
            0,
        )

    def save_observations(
        self, obs, folder_name, method_name: str = "grounded_mobile_sam"
    ):
        """
        Save the observations to the folder
        """
        object_seg = obs.semantic == 1

        os.makedirs(os.path.join(folder_name, method_name), exist_ok=True)
        # save object segmentation
        cv2.imwrite(
            os.path.join(folder_name, method_name, "obj.png"),
            object_seg.astype(np.uint8) * 255,
        )

        recep_seg = obs.semantic == 2

        # save receptacle segmentation
        cv2.imwrite(
            os.path.join(folder_name, method_name, "rec.png"),
            recep_seg.astype(np.uint8) * 255,
        )

    def load_segmentations_for_method(
        self, folder_name: str, method_name: str = "grounded_mobile_sam"
    ):
        obj_seg = (
            cv2.imread(
                os.path.join(folder_name, method_name, "obj.png"), cv2.IMREAD_GRAYSCALE
            )
            // 255
        )
        recep_seg = (
            cv2.imread(
                os.path.join(folder_name, method_name, "rec.png"), cv2.IMREAD_GRAYSCALE
            )
            // 255
        )
        return obj_seg, recep_seg

    def quantitative_evaluate(
        self, folder_name: str, method_name: str = "grounded_mobile_sam"
    ):
        (
            rgb,
            gt_object_mask,
            gt_recep_mask,
            object_name,
            start_recep,
        ) = self.read_images_from_folder(folder_name)
        obj_seg, recep_seg = self.load_segmentations_for_method(
            folder_name, method_name=method_name
        )
        # compare IoU of object segmentation with gt_object_mask
        object_iou = self.compute_iou(gt_object_mask, obj_seg)
        recep_iou = self.compute_iou(gt_recep_mask, recep_seg)
        # return a dataframe
        return pd.DataFrame(
            {
                "object_name": object_name,
                "start_recep": start_recep,
                "object_iou": object_iou,
                "recep_iou": recep_iou,
                "folder_name": folder_name,
                "object_detected?": object_iou > 0.5,
                "recep_detected?": recep_iou > 0.5,
                "false_object": np.sum(obj_seg * (1 - gt_object_mask)),
                "false_recep": np.sum(recep_seg * (1 - gt_recep_mask)),
            },
            index=["folder_name"],
        )

    def run_single_evaluation(
        self,
        folder_name: str,
        method_name: str = "grounded_mobile_sam",
        save_observations=False,
    ):
        """
        1. Load image, groundtruth segmentation
        2. Generate segmentation for image
        3. Evaluate segmentation
        4. Save segmentation
        5. Return evaluation results
        """
        # if True:
        try:
            (
                rgb,
                gt_object_mask,
                gt_recep_mask,
                object_name,
                start_recep,
            ) = self.read_images_from_folder(folder_name)
            obs = Observations(gps=None, compass=None, depth=None, rgb=rgb)
            self.reset_vocab(["", object_name, start_recep])
            if save_observations:
                # predict object segmentation
                obs = self.predict(obs)
                self.save_observations(obs, folder_name, method_name=method_name)
            return self.quantitative_evaluate(folder_name, method_name=method_name)

            # # evaluate object segmentation
            # object_iou = self.evaluate(obs.semantic, gt_object_mask, gt_recep_mask)
        except Exception as e:
            print(f"Error: {e} for folder {folder_name}")
            return None

    def evaluate_folder(
        self, folder_name: str, method_name: str = "grounded_mobile_sam"
    ):
        time_start = time.time()
        processed = 0
        dfs = []
        for sub_folder in os.listdir(folder_name):
            success = self.run_single_evaluation(
                os.path.join(folder_name, sub_folder), method_name=method_name
            )
            time_end = time.time()
            if success is not None:
                dfs.append(success)
            processed += success is not None
        if processed > 0:
            print(f"Time per image: {(time_end - time_start) / processed}")

        df = pd.concat(dfs)
        df.to_csv(os.path.join(folder_name, f"{method_name}.csv"))
        print(
            "object_iou",
            "recep_iou",
            "object_detected?",
            "recep_detected?",
            "false_object_pixels",
            "false_recep_pixels",
        )
        print(
            df["object_iou"].mean(),
            df["recep_iou"].mean(),
            df["object_detected?"].mean(),
            df["recep_detected?"].mean(),
            df["false_object"].mean(),
            df["false_recep"].mean(),
        )
        return df


if __name__ == "__main__":
    perception_module = GroundedMobileSAM()
    perception_module.evaluate_folder(
        "/srv/flash1/syenamandra3/object-rearrangement/ovmm/home-robot/benchmarks/angle_52_ternary/",
        method_name="grounded_mobile_sam_unary",
    )
    perception_module.evaluate_folder(
        "/srv/flash1/syenamandra3/object-rearrangement/ovmm/home-robot/benchmarks/angle_52_ternary/",
        method_name="grounded_mobile_sam_binary",
    )
    perception_module.evaluate_folder(
        "/srv/flash1/syenamandra3/object-rearrangement/ovmm/home-robot/benchmarks/angle_52_ternary/",
        method_name="detic_ternary",
    )
