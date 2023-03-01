from typing import Tuple
import cv2
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
import skimage.morphology
from omegaconf import DictConfig

import home_robot.utils.pose as pu
from home_robot.core.interfaces import Observations

from home_robot.perception.detection.detic.detic_mask import Detic
from .superglue import Matching


class ObsPreprocessor:

    def __init__(self, config: DictConfig, device: torch.device) -> None:
        self.device = device
        self.frame_height = config.frame_height
        self.frame_width = config.frame_width

        self.depth_filtering = config.semantic_prediction.depth_filtering
        self.depth_filter_range_cm = (
            config.semantic_prediction.depth_filter_range_cm
        )
        self.preprojection_kp_dilation = config.preprojection_kp_dilation
        self.match_projection_threshold = (
            config.superglue.match_projection_threshold
        )

        #  initialize detection and localization modules
        self.matching = Matching(
            device=config.simulator_gpu_id,
            config=config.superglue,
            default_vis_dir=f"{config.dump_location}/images/{config.exp_name}",
            print_images=config.generate_videos,
        )
        self.instance_seg = Detic(config.detic)

        # init episode variables
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = None
        self.step = None

    def reset(self):
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = np.zeros(3)
        self.step = 0

    def preprocess(self, obs: Observations) -> Tuple[Tensor, ndarray, ndarray]:
        """
        Preprocess observations of a single timestep batched across
        environments.

        Arguments:
            obs: list of observations of length num_environments

        Returns:
            obs_preprocessed: frame containing (RGB, depth, keypoint_loc) of
               shape (3 + 1 + 1, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
               of shape (num_environments, 3)
            matches: keypoint correspondences from goal image to egocentric
               image of shape (1, n)
            confidence: keypoint correspondence confidence of shape (1, n)
        """
        if self.goal_image is None:
            img_goal = obs.task_observations["instance_imagegoal"]
            (
                self.goal_image, self.goal_image_keypoints
            ) = self.matching.get_goal_image_keypoints(img_goal)
            self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

        pose_delta, self.last_pose = self._preprocess_pose(obs)
        obs_preprocessed, matches, confidence = self._preprocess_frame(obs)

        self.step += 1
        return obs_preprocessed, pose_delta, matches, confidence

    def _preprocess_frame(
        self, obs: Observations
    ) -> Tuple[Tensor, ndarray, ndarray]:
        """Preprocess frame information in the observation."""

        def downscale(rgb: ndarray, depth: ndarray):
            ds = rgb.shape[1] / self.frame_width
            if ds == 1:
                return rgb, depth
            dim = (self.frame_width, self.frame_height)
            rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)[:, :, None]
            return rgb, depth

        def preprocess_keypoint_localization(
            rgb, goal_keypoints, rgb_keypoints, matches, confidence
        ):
            goal_keypoints = goal_keypoints[0].cpu().to(dtype=int).numpy()
            rgb_keypoints = rgb_keypoints[0].cpu().to(dtype=int).numpy()
            confidence = confidence[0]
            matches = matches[0]

            # map the valid goal keypoints to ego keypoints
            is_in_mask = self.goal_mask[goal_keypoints[:, 1], goal_keypoints[:, 0]]
            has_high_confidence = confidence >= self.match_projection_threshold
            is_matching_kp = matches > -1
            valid = np.logical_and(is_in_mask, has_high_confidence, is_matching_kp)
            matched_rgb_keypoints = rgb_keypoints[matches[valid]]

            # set matched rgb keypoints as goal points
            kp_loc = np.zeros((*rgb.shape[:2], 1), dtype=rgb.dtype)
            kp_loc[matched_rgb_keypoints[:, 1], matched_rgb_keypoints[:, 0]] = 1

            if self.preprojection_kp_dilation > 0:
                disk = skimage.morphology.disk(self.preprojection_kp_dilation)
                kp_loc = np.expand_dims(
                    cv2.dilate(kp_loc, disk, iterations=1), axis=2
                )

            return kp_loc

        depth = np.expand_dims(obs.depth, axis=2) * 100.0
        rgb, depth = downscale(obs.rgb, depth)

        (
            goal_keypoints, rgb_keypoints, matches, confidence
        ) = self.matching(
            rgb,
            goal_image=self.goal_image,
            goal_image_keypoints=self.goal_image_keypoints,
            step=self.step,
        )
        kp_loc = preprocess_keypoint_localization(
            rgb, goal_keypoints, rgb_keypoints, matches, confidence
        )

        obs_preprocessed = np.concatenate([rgb, depth, kp_loc], axis=2)
        obs_preprocessed = obs_preprocessed.transpose(2, 0, 1)
        obs_preprocessed = torch.from_numpy(obs_preprocessed)
        obs_preprocessed = obs_preprocessed.to(device=self.device)
        obs_preprocessed = obs_preprocessed.unsqueeze(0)
        return obs_preprocessed, matches, confidence

    def _preprocess_pose(self, obs: Observations) -> Tuple[Tensor, ndarray]:
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose)
        ).unsqueeze(0).to(device=self.device)
        return pose_delta, curr_pose
