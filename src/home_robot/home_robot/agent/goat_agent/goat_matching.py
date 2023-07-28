# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import clip
import matplotlib
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)
from tqdm import tqdm

from home_robot.agent.imagenav_agent.superglue import Matching
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

matplotlib.use("Agg")


class GoatMatching(Matching):
    def __init__(
        self,
        device: int,
        score_func: str,
        score_thresh: float,
        num_sem_categories: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        assert score_func in ["confidence_sum", "match_count"]
        self.score_func = score_func
        self.score_thresh = score_thresh
        self.num_sem_categories = num_sem_categories

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device)
        n_px = self.clip_model.visual.input_resolution
        self.clip_image_preprocess = Compose(
            [
                Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def get_matches_against_memory(
        self,
        instance_memory: InstanceMemory,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        in the instance memory.
        """
        all_matches, all_confidences = [], []
        instances = instance_memory.instance_views[0]
        all_views = []
        instance_view_counts = []
        steps_per_view = []
        for (inst_key, inst) in tqdm(
            instances.items(), desc="Matching goal image with instance views"
        ):
            inst_views = inst.instance_views
            for view_idx, inst_view in enumerate(inst_views):
                # if inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1] < 2500 or (np.array(inst_view.cropped_image.shape[0:2]) < 15).any():
                #     continue
                img = instance_memory.images[0][inst_view.timestep].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                all_views.append(img)
                steps_per_view.append(1000 * step + 10 * inst_key + view_idx)
            instance_view_counts.append(len(inst_views))

        if len(all_views) > 0:
            if image_goal is not None:
                _, _, all_matches, all_confidences = matching_fn(
                    all_views,
                    goal_image=image_goal,
                    goal_image_keypoints=kwargs["goal_image_keypoints"],
                    step=1000 * step + 10 * inst_key + view_idx,
                )
            elif language_goal is not None:
                all_matches, all_confidences = matching_fn(
                    all_views,
                    language_goal,
                    step=1000 * step + 10 * inst_key + view_idx,
                )

            # unflatten based on number of views per instance
            all_matches = np.concatenate(all_matches, 0)
            all_confidences = np.concatenate(all_confidences, 0)
            all_matches = np.split(all_matches, np.cumsum(instance_view_counts)[:-1])
            all_confidences = np.split(
                all_confidences, np.cumsum(instance_view_counts)[:-1]
            )
            return all_matches, all_confidences
        return [], []

    @torch.no_grad()
    def match_image_to_image(
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
                rgb_image_processed = self._preprocess_image(
                    rgb_image_batched[i].astype(np.uint8)
                )
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

    @torch.no_grad()
    def match_language_to_image(self, views, language_goal, **kwargs):
        """Compute matching scores from a language goal to images."""
        batch_size = 64
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size
        if isinstance(views, list):
            views = np.stack(views, 0)
        if views.dtype == np.uint8:
            views = views.astype(np.float32) / 255
        views = torch.cat(
            [
                self.clip_model.encode_image(
                    self.clip_image_preprocess(v.permute(0, 3, 1, 2)).to(self.device)
                )
                for v in torch.tensor(views).split(batch_size)
            ],
            dim=0,
        )
        # compute similarity
        similarity = (language_goal @ views.T).softmax(dim=-1)
        return [[1]] * similarity.shape[0], np.expand_dims(
            similarity.detach().cpu().numpy(), 1
        )

    def select_and_localize_instance(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
        instance_goal_found: bool,
        goal_inst: Optional[int],
        all_matches: List = None,
        all_confidences: List = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[int]]:
        """Select and localize an instance given computed matching scores."""

        if all_matches is not None:
            if len(all_matches) > 0:
                max_scores = []
                for inst_idx, match_inst in enumerate(all_matches):
                    inst_view_scores = []
                    for view_idx, match_view in enumerate(match_inst):
                        view_score = all_confidences[inst_idx][view_idx][
                            match_view != -1
                        ].sum()
                        inst_view_scores.append(view_score)

                    max_scores.append(max(inst_view_scores))
                    print(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")

                if max(max_scores) > self.score_thresh:
                    inst_idx = np.argmax(max_scores)
                    instance_map = local_map[0][
                        MC.NON_SEM_CHANNELS
                        + self.num_sem_categories : MC.NON_SEM_CHANNELS
                        + 2 * self.num_sem_categories,
                        :,
                        :,
                    ]  # TODO: currently assuming img goal instance was an object outside of the vocabulary
                    inst_map_idx = instance_map == inst_idx + 1
                    inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                    goal_map_temp = (instance_map[inst_map_idx] == inst_idx + 1).to(
                        torch.float
                    )

                    if goal_map_temp.any():
                        instance_goal_found = True
                        goal_inst = inst_idx + 1
                        goal_map = goal_map_temp
                        print(f"{goal_inst} will be the goal")
                    else:
                        print("Instance was seen, but not present in local map.")
                else:
                    print("Goal image does not match any instance.")
                    # TODO: dont stop at the first instance, but rather find the best one

        if goal_inst is not None and instance_goal_found is True:
            found_goal[0] = True

            instance_map = local_map[0][
                MC.NON_SEM_CHANNELS
                + self.num_sem_categories : MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
                :,
                :,
            ]
            inst_map_idx = instance_map == goal_inst
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
            goal_map = (instance_map[inst_map_idx] == goal_inst).to(torch.float)

        else:
            for e in range(confidence.shape[0]):
                # if the goal category is empty, the goal can't be found
                if not local_map[e, 21].any().item():
                    continue

                if self.score_func == "confidence_sum":
                    score = confidence[e][matches[e] != -1].sum()
                else:  # match_count
                    score = (matches[e] != -1).sum()

                if score < self.score_thresh:
                    continue

                found_goal[e] = True
                # Set goal_map to the last channel of the local semantic map
                goal_map[e, 0] = local_map[e, 21]

        return goal_map, found_goal, instance_goal_found, goal_inst
