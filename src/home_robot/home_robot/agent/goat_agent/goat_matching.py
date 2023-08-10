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
MIN_PIXELS = 1000
MIN_EDGE = 15


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

    def get_matches_against_current_frame(
        self,
        instance_memory,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        use_full_image=None,
        categories=None,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        detected in the current frame.
        """
        # TODO We should restrict detections in the current frame by category
        detections = []
        instance_ids = []
        # first collect crops of instances found in the current frame
        for local_instance_id, inst_view in instance_memory.unprocessed_views[
            0
        ].items():
            if categories is not None and inst_view.category_id not in categories:
                continue
            if (
                inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1]
                < MIN_PIXELS
                or (np.array(inst_view.cropped_image.shape[0:2]) < MIN_EDGE).any()
            ):
                continue
            if use_full_image:
                img = instance_memory.images[0][-1]
            else:
                img = inst_view.cropped_image
            detections.append(img)
            instance_ids.append(local_instance_id)

        matches, confidences = [], []
        if len(detections) > 0:
            matches, confidences = self.match_images_to_goal(
                detections,
                matching_fn,
                step,
                image_goal=image_goal,
                language_goal=language_goal,
                **kwargs,
            )
        return np.array([matches]), np.array([confidences]), np.array([instance_ids])

    def match_images_to_goal(
        self,
        all_views,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        **kwargs,
    ):
        all_matches, all_confidences = [], []
        if image_goal is not None:
            _, _, all_matches, all_confidences = matching_fn(
                all_views,
                goal_image=image_goal,
                goal_image_keypoints=kwargs["goal_image_keypoints"],
                step=1000 * step,
            )
        elif language_goal is not None:
            all_matches, all_confidences = matching_fn(
                all_views,
                language_goal,
            )
        return all_matches, all_confidences

    def get_matches_against_memory(
        self,
        instance_memory: InstanceMemory,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        use_full_image=False,
        categories=None,
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
        instance_ids = []
        for (inst_key, inst) in instances.items():
            if categories is not None and inst.category_id not in categories:
                continue
            inst_views = inst.instance_views
            views_added = 0
            for view_idx, inst_view in enumerate(inst_views):
                if (
                    inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1]
                    < MIN_PIXELS
                    or (np.array(inst_view.cropped_image.shape[0:2]) < MIN_EDGE).any()
                ):
                    continue
                if use_full_image:
                    img = instance_memory.images[0][inst_view.timestep].cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))
                else:
                    img = inst_view.cropped_image

                all_views.append(img)
                views_added += 1
                steps_per_view.append(1000 * step + 10 * inst_key + view_idx)
            if views_added > 0:
                instance_view_counts.append(views_added)
                instance_ids.append(inst_key)

        if len(all_views) > 0:
            all_matches, all_confidences = self.match_images_to_goal(
                all_views,
                matching_fn,
                step,
                image_goal=image_goal,
                language_goal=language_goal,
                **kwargs,
            )
            # unflatten based on number of views per instance
            all_matches = np.concatenate(all_matches, 0)
            all_confidences = np.concatenate(all_confidences, 0)
            all_matches = np.split(all_matches, np.cumsum(instance_view_counts)[:-1])
            all_confidences = np.split(
                all_confidences, np.cumsum(instance_view_counts)[:-1]
            )
            return all_matches, all_confidences, instance_ids
        return [], [], []

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
            self._visualize(matcher_inputs, pred, step + i)

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

    def get_best_match(self, scores, instance_ids, instance_map, score_thresh):
        instance_goal_found = False
        goal_inst = None
        sorted_inst_ids = np.argsort(scores)[::-1]
        idx = 0
        while (
            idx < len(sorted_inst_ids) and scores[sorted_inst_ids[idx]] > score_thresh
        ):
            inst_idx = sorted_inst_ids[idx]
            idx += 1
            print(
                f"Trying to localize instance {inst_idx + 1} with score {scores[inst_idx]}"
            )
            if instance_ids is None:
                best_instance_id = inst_idx + 1
            else:
                best_instance_id = instance_ids[inst_idx]
            if instance_ids[inst_idx] == -1:
                print("instance_ids[inst_idx] == -1")
                continue
            inst_map_idx = instance_map == best_instance_id
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))

            goal_map_temp = (instance_map[inst_map_idx] == best_instance_id).float()

            if goal_map_temp.any():
                instance_goal_found = True
                goal_inst = best_instance_id
                print(f"Instance {goal_inst} will be the goal")
                return instance_goal_found, goal_inst
            else:
                print("Instance was seen, but not present in local map.")

        if idx == len(sorted_inst_ids):
            print("Goal image does not match any instance.")

        return instance_goal_found, goal_inst

    def aggregate_scores_per_instance(self, matches, confidences, agg_fn):
        agg_scores = []
        if len(matches) > 0:
            for inst_idx, match_inst in enumerate(matches):
                inst_view_scores = []
                for view_idx, match_view in enumerate(match_inst):
                    view_score = confidences[inst_idx][view_idx][match_view != -1].sum()
                    inst_view_scores.append(view_score)

                if agg_fn == "max":
                    agg_scores.append(max(inst_view_scores))
                elif agg_fn == "mean":
                    agg_scores.append(np.mean(inst_view_scores))
                elif agg_fn == "median":
                    agg_scores.append(np.median(inst_view_scores))
                else:
                    raise NotImplementedError
                print(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")
        return agg_scores

    def get_goal_map_from_goal_instance(
        self, instance_map, goal_map, goal_inst, instance_goal_found, found_goal
    ):
        if goal_inst is not None and instance_goal_found is True:
            found_goal[0] = True
            inst_map_idx = instance_map == goal_inst
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
            goal_map = (instance_map[inst_map_idx] == goal_inst).to(torch.float)
        return goal_map, found_goal

    def select_and_localize_instance(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
        local_instance_ids: List,
        local_id_to_global_id_map: Optional[Dict],
        instance_goal_found: bool,
        goal_inst: Optional[int],
        all_matches: List = None,
        all_confidences: List = None,
        instance_ids: List = None,
        score_thresh: float = 0.0,
        agg_fn: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[int]]:
        """Select and localize an instance given computed matching scores."""
        print(f"Selecting and localizing an instance with threshold {score_thresh}")
        instance_map = local_map[0][
            MC.NON_SEM_CHANNELS
            + self.num_sem_categories : MC.NON_SEM_CHANNELS
            + 2 * self.num_sem_categories,
            :,
            :,
        ]

        if goal_inst is not None and instance_goal_found is True:
            goal_map, found_goal = self.get_goal_map_from_goal_instance(
                instance_map, goal_map, goal_inst, instance_goal_found, found_goal
            )
            return goal_map, found_goal, instance_goal_found, goal_inst

        if all_matches is not None:
            if len(all_matches) > 0:
                agg_scores = self.aggregate_scores_per_instance(
                    all_matches, all_confidences, agg_fn
                )
                if len(agg_scores) > 0:
                    instance_goal_found, goal_inst = self.get_best_match(
                        agg_scores, instance_ids, instance_map, score_thresh
                    )

        if goal_inst is None and matches is not None:
            for e in range(confidence.shape[0]):
                scores = confidence[e]

                if len(scores) > 0:
                    global_instance_ids = [
                        local_id_to_global_id_map[e].get(i, -1)
                        for i in local_instance_ids[e]
                    ]
                    agg_scores = self.aggregate_scores_per_instance(
                        matches[e], confidence[e], agg_fn
                    )
                    instance_goal_found, goal_inst = self.get_best_match(
                        agg_scores, global_instance_ids, instance_map, score_thresh
                    )

        if goal_inst is not None and instance_goal_found is True:
            goal_map, found_goal = self.get_goal_map_from_goal_instance(
                instance_map, goal_map, goal_inst, instance_goal_found, found_goal
            )

        return goal_map, found_goal, instance_goal_found, goal_inst