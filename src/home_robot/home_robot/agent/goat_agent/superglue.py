# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from home_robot.agent.imagenav_agent.SuperGluePretrainedNetwork.models.matching import (
    Matching as SGPMatching,
)
from home_robot.agent.imagenav_agent.superglue import Matching

matplotlib.use("Agg")


class GOATMatching(Matching):
    """ " Implement matching between images"""

    def __init__(
        self,
        device: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

    @torch.no_grad()
    def forward(
        self,
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        rgb_image_keypoints: Optional[Dict[str, Any]] = None,
        goal_image_keypoints: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ):
        """Computes and describes keypoints using SuperPoint and matches
        keypoints between an RGB image and a goal image using SuperGlue.
        Either goal_image or goal_image_keypoints must be provided.
        Returns:
            tensor of goal image keypoints
            tensor of rgb image keypoints
            tensor of keypoint matches
            tensor of match confidences
        """
        if isinstance(rgb_image, np.ndarray) and len(rgb_image.shape) == 3:
            rgb_image_batched = [rgb_image]
        else:
            rgb_image_batched = rgb_image
            assert rgb_image_keypoints is None

        all_goal_keypoints = []
        all_rgb_keypoints = []
        all_matches = []
        all_confidences = []
        for i in range(len(rgb_image_batched)):
            if goal_image_keypoints is None:
                goal_image_keypoints = {}
            if rgb_image_keypoints is None:
                rgb_image_keypoints = {}

            if isinstance(goal_image, np.ndarray):
                goal_image_processed = self._preprocess_image(goal_image)
            else:
                goal_image_processed = goal_image
            if isinstance(rgb_image_batched[i], np.ndarray):
                rgb_image_processed = self._preprocess_image(rgb_image_batched[i])
            else:
                rgb_image_processed = rgb_image_batched[i]

            matcher_inputs = {
                "image0": goal_image_processed,
                "image1": rgb_image_processed,
                **goal_image_keypoints,
                **rgb_image_keypoints,
            }
            pred = self.matcher(matcher_inputs)
            matches = pred["matches0"].cpu().numpy()
            confidence = pred["matching_scores0"].cpu().numpy()
            self._visualize(matcher_inputs, pred, step)

            if "keypoints0" in matcher_inputs:
                goal_keypoints = matcher_inputs["keypoints0"]
            else:
                goal_keypoints = pred["keypoints0"]

            if "keypoints1" in matcher_inputs:
                rgb_keypoints = matcher_inputs["keypoints1"]
            else:
                rgb_keypoints = pred["keypoints1"]
            if isinstance(rgb_image, np.ndarray) and len(rgb_image.shape) == 3:
                return goal_keypoints, rgb_keypoints, matches, confidence

            all_goal_keypoints.append(goal_keypoints)
            all_rgb_keypoints.append(rgb_keypoints)
            all_matches.append(matches)
            all_confidences.append(confidence)
        return all_goal_keypoints, all_rgb_keypoints, all_matches, all_confidences
