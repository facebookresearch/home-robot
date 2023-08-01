# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import clip
import matplotlib
import numpy as np
import torch
from torchvision.transforms import ToPILImage
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
        num_sem_categories: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        assert score_func in ["confidence_sum", "match_count"]
        self.score_func = score_func
        self.num_sem_categories = num_sem_categories

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device)

    def get_matches_against_memory(
        self,
        instance_memory: InstanceMemory,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        use_full_image=False,
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
        for (inst_key, inst) in instances.items():
            inst_views = inst.instance_views
            for view_idx, inst_view in enumerate(inst_views):
                # if inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1] < 2500 or (np.array(inst_view.cropped_image.shape[0:2]) < 15).any():
                #     continue
                if use_full_image:
                    img = instance_memory.images[0][inst_view.timestep].cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))
                else:
                    img = inst_view.cropped_image
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

        # TODO Can we batch this for loop to speed it up? It is a bottleneck
        print("Computing matching score with each view...")
        for i in tqdm(range(len(rgb_image_batched))):
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
    def match_language_to_image(self, views_orig, language_goal, **kwargs):
        """Compute matching scores from a language goal to images."""
        batch_size = 64
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size

        views = views_orig
        views = torch.stack(
            [self.clip_preprocess(ToPILImage()(v.astype(np.uint8))) for v in views],
            dim=0,
        )
        view_embeddings = torch.cat(
            [
                self.clip_model.encode_image(v.to(self.device))
                for v in views.split(batch_size)
            ],
            dim=0,
        )
        # normalize the embeddings
        view_embeddings = view_embeddings / view_embeddings.norm(dim=-1, keepdim=True)
        language_goal = language_goal / language_goal.norm(dim=-1, keepdim=True)
        # compute cosines similarity
        similarity = (language_goal @ view_embeddings.T).squeeze(0)
        return [[[1]]] * similarity.shape[0], similarity.detach().cpu().numpy().reshape(
            -1, 1, 1
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
        score_thresh: float = 0.0,
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

                    # Take the max matching score across views of the instance
                    # TODO Try other aggregation strategies (mean, median)
                    max_scores.append(max(inst_view_scores))
                    print(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")

                sorted_inst_ids = np.argsort(max_scores)[::-1]
                idx = 0
                while (
                    idx < len(sorted_inst_ids)
                    and max_scores[sorted_inst_ids[idx]] > score_thresh
                ):
                    inst_idx = sorted_inst_ids[idx]
                    instance_map = local_map[0][
                        MC.NON_SEM_CHANNELS
                        + self.num_sem_categories : MC.NON_SEM_CHANNELS
                        + 2 * self.num_sem_categories,
                        :,
                        :,
                    ]
                    inst_map_idx = instance_map == inst_idx + 1
                    inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                    goal_map_temp = (instance_map[inst_map_idx] == inst_idx + 1).float()

                    breakpoint()
                    if goal_map_temp.any():
                        instance_goal_found = True
                        goal_inst = inst_idx + 1
                        goal_map = goal_map_temp
                        print(f"{goal_inst} will be the goal")
                        break
                    else:
                        print("Instance was seen, but not present in local map.")
                    idx += 1

                if idx == len(sorted_inst_ids):
                    print("Goal image does not match any instance.")

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

        elif confidence is not None and matches is not None:
            for e in range(confidence.shape[0]):
                # if the goal category is empty, the goal can't be found
                if not local_map[e, 21].any().item():
                    continue

                if self.score_func == "confidence_sum":
                    score = confidence[e][matches[e] != -1].sum()
                else:  # match_count
                    score = (matches[e] != -1).sum()

                if score < score_thresh:
                    continue

                found_goal[e] = True
                # Set goal_map to the last channel of the local semantic map
                goal_map[e, 0] = local_map[e, 21]

        return goal_map, found_goal, instance_goal_found, goal_inst
