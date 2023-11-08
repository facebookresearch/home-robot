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
        instance_memory: InstanceMemory,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        assert score_func in ["confidence_sum", "match_count"]
        self.score_func = score_func
        self.num_sem_categories = num_sem_categories

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device)
        self.goto_past_pose = config.goto_past_pose
        self.instance_memory = instance_memory

    def get_matches_against_current_frame(
        self,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        cropping_mode='full', # full, bbox, padded_bbox_%d
        categories=None,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        detected in the current frame.
        """
        instance_memory = self.instance_memory
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
            img = self.get_cropped_image(inst_view, cropping_mode=cropping_mode)
            detections.append(img)
            instance_ids.append(local_instance_id)

        matches, confidences = [], []
        if len(detections) > 0:
            matches, confidences = self.match_images_to_goal(
                detections,
                matching_fn,
                step,
                use_full_image=cropping_mode == 'full',
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
        use_full_image=False,
        image_goal=None,
        language_goal=None,
        matching_method='superglue',
        **kwargs,
    ):
        all_matches, all_confidences = [], []
        if image_goal is not None:
            all_matches, all_confidences = matching_fn(
                all_views,
                goal_image=image_goal,
                use_full_image=use_full_image,
                step=1000 * step,
                matching_method=matching_method,
                **kwargs,
            )
        elif language_goal is not None:
            all_matches, all_confidences = matching_fn(
                all_views, language_goal, **kwargs
            )
        return all_matches, all_confidences

    def get_cropped_image(self, inst_view, cropping_mode='full'):
        instance_memory = self.instance_memory
        full_img = instance_memory.images[0][inst_view.timestep].cpu().numpy()
        full_img = np.transpose(full_img, (1, 2, 0))
        if cropping_mode == 'full':
            return full_img
        elif 'bbox' in cropping_mode:
            p = 0
            if 'padded' in cropping_mode:
                p = int(cropping_mode.split('_')[-1])
            else: assert cropping_mode == 'bbox'
            bbox = inst_view.bbox
            # get bounding box
            h, w = full_img.shape[:2]
            cropped_image = full_img[
                max(bbox[0, 0] - p, 0) : min(bbox[1, 0] + p, h),
                max(bbox[0, 1] - p, 0) : min(bbox[1, 1] + p, w),
            ]
            return cropped_image
        else:
            raise ValueError(f"Invalid cropping mode: {cropping_mode}")

    def get_matches_against_memory(
        self,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        cropping_mode='full', # full, bbox, padded_bbox_%d
        categories=None,
        aggregate_feats=False,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        in the instance memory.
        """
        instance_memory = self.instance_memory
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
                cropped_image = self.get_cropped_image(inst_view, cropping_mode=cropping_mode)
                all_views.append(cropped_image)
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
                use_full_image=cropping_mode == 'full',
                image_goal=image_goal,
                language_goal=language_goal,
                instance_view_counts=instance_view_counts,
                aggregate_feats=aggregate_feats,
                **kwargs,
            )
            if aggregate_feats:
                instance_view_counts = np.ones(
                    len(instance_view_counts), dtype=np.int64
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


    def match_image_to_image(
        self,
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        rgb_image_keypoints: Optional[Dict[str, Any]] = None,
        goal_image_keypoints: Optional[Dict[str, Any]] = None,
        use_full_image: bool = False,
        step: Optional[int] = None,
        matching_method: str = "clip",
        **kwargs,
        
    ):
        if matching_method == "superglue":
            return self.match_image_to_image_superglue(rgb_image, goal_image, rgb_image_keypoints, goal_image_keypoints, use_full_image, step, **kwargs)
        elif matching_method == "clip":
            return self.match_image_to_image_clip(rgb_image, goal_image, **kwargs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def match_image_to_image_clip(
        self,
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):

        if isinstance(goal_image, torch.Tensor):
            goal_image = goal_image.detach().cpu().numpy()

        if len(goal_image.shape) == 3:
            goal_image = np.expand_dims(goal_image, 0)

        view_embeddings = self.get_image_embeddings(rgb_image, **kwargs)
        goal_embeddings = self.get_image_embeddings(goal_image)

        # compute cosines similarity
        similarity = (goal_embeddings @ view_embeddings.T).squeeze(0)
        return [[[1]]] * similarity.shape[0], similarity.detach().cpu().numpy().reshape(
            -1, 1, 1
        )

    @torch.no_grad()
    def match_image_to_image_superglue(
        self,
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        rgb_image_keypoints: Optional[Dict[str, Any]] = None,
        goal_image_keypoints: Optional[Dict[str, Any]] = None,
        use_full_image: bool = False,
        step: Optional[int] = None,
        **kwargs,
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
        if kwargs.get("aggregate_feats", False):
            raise NotImplementedError

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
                if rgb_image_batched[i].shape[0] == 3:
                    rgb_image_batched[i] = rgb_image_batched[i].transpose(1, 2, 0)
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
        return all_matches, all_confidences

    @torch.no_grad()
    def match_image_batch_to_image(
        self,
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        rgb_image_keypoints: Optional[Dict[str, Any]] = None,
        goal_image_keypoints: Optional[Dict[str, Any]] = None,
        use_full_image: bool = False,
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

        if use_full_image is not True:
            """
            add empty zero padding around instance crops to
            make them all the same size so they can be batched
            """
            padded_detections = []
            max_detection_w = max([x.shape[0] for x in rgb_image])
            max_detection_h = max([x.shape[1] for x in rgb_image])
            padding_bg = (
                np.zeros((max_detection_w, max_detection_h, 3), dtype=np.uint8) * 255
            )
            for detection in rgb_image:
                w = detection.shape[0]
                h = detection.shape[1]
                padding_bg_new = padding_bg.copy()
                padding_bg_new[:w, :h, :] = detection
                padded_detections.append(padding_bg_new)

            rgb_image = padded_detections

        if isinstance(rgb_image, np.ndarray) and len(rgb_image.shape) == 3:
            rgb_image_batched = [rgb_image]
        else:
            rgb_image_batched = rgb_image
            assert rgb_image_keypoints is None

        # TODO Can we batch this for loop to speed it up? It is a bottleneck
        print("Computing matching score with each view...")

        if isinstance(goal_image, np.ndarray):
            goal_image_processed = self._preprocess_image(goal_image)
        else:
            goal_image_processed = goal_image

        for i in range(len(rgb_image_batched)):
            if rgb_image_batched[i].shape[0] == 3:
                rgb_image_batched[i] = rgb_image_batched[i].transpose(1, 2, 0)
            rgb_image_batched[i] = self._preprocess_image(
                rgb_image_batched[i].astype(np.uint8)
            )

        if goal_image_keypoints is None:
            goal_image_keypoints = {}
        if rgb_image_keypoints is None:
            rgb_image_keypoints = {}

        matcher_inputs = {
            "image0": goal_image_processed,
            "image1": rgb_image_batched,
            **goal_image_keypoints,
            **rgb_image_keypoints,
        }
        pred = self.matcher(matcher_inputs)
        matches = pred["matches0"].cpu().numpy()
        confidence = pred["matching_scores0"].cpu().numpy()
        for i in range(len(rgb_image_batched)):
            single_matcher_input = {
                "image0": goal_image_processed,
                "image1": rgb_image_batched[i],
                **goal_image_keypoints,
                **rgb_image_keypoints,
            }

            self._batched_visualize(single_matcher_input, pred, step + i, idx=i)

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

        confidence = confidence[:, np.newaxis, :]
        matches = matches[:, np.newaxis, :]

        return goal_keypoints, rgb_keypoints, matches.tolist(), confidence.tolist()


    def get_image_embeddings(
        self, 
        views_orig, 
        aggregate_feats=False,
        feat_agg_fn="mean",
        instance_view_counts=None,
        **kwargs,
    ):
        batch_size = 64
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

        if aggregate_feats:
            assert instance_view_counts is not None
            view_embeddings_grouped = np.split(
                view_embeddings, np.cumsum(instance_view_counts)[:-1]
            )
            if feat_agg_fn == "mean":
                view_embeddings = torch.stack(
                    [torch.mean(views, dim=0) for views in view_embeddings_grouped],
                    dim=0,
                )
            else:
                raise NotImplementedError
        return view_embeddings

    @torch.no_grad()
    def match_language_to_image(self, views_orig, language_goal, **kwargs):
        """Compute matching scores from a language goal to images."""
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size

        language_goal = language_goal / language_goal.norm(dim=-1, keepdim=True)
        view_embeddings = self.get_image_embeddings(views_orig, **kwargs)
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

            if not self.goto_past_pose:
                goal_map_temp = (instance_map[inst_map_idx] == best_instance_id).float()
                if goal_map_temp.any():
                    instance_goal_found = True
                    goal_inst = best_instance_id
                    print(f"Instance {goal_inst} will be the goal")
                    return instance_goal_found, goal_inst
                else:
                    print("Instance was seen, but not present in local map.")
            else:
                # we are ok with object not being on map when using agent pose as target
                return True, best_instance_id

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
                elif "top" in agg_fn:
                    top_k, np_agg_fn = agg_fn.split("_")[-2:]
                    assert np_agg_fn in ["mean", "median"]
                    top_k = int(top_k)
                    inst_view_scores.sort(reverse=True)
                    agg_scores.append(getattr(np, np_agg_fn)(inst_view_scores[:top_k]))
                else:
                    raise NotImplementedError
                print(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")
        return agg_scores

    def get_goal_map_from_goal_instance(
        self, instance_map, goal_map, lmb, goal_inst, instance_goal_found, found_goal
    ):
        goal_pose = None
        if goal_inst is not None and instance_goal_found is True:
            found_goal[0] = True
            if self.goto_past_pose:
                instance_memory = self.instance_memory
                instance_views = instance_memory.instance_views[0][
                    goal_inst
                ].instance_views
                # pick a view with maximum object coverage
                best_view = np.argmax([view.object_coverage for view in instance_views])
                pose = instance_views[best_view].pose
                curr_x, curr_y, curr_o, gy1, _, gx1, _ = pose.tolist()
                goal_map = torch.zeros(instance_map[0].shape)
                pos = (
                    int(curr_x * 100.0 / 5 - lmb[0][2]),
                    int(curr_y * 100.0 / 5 - lmb[0][0]),
                )
                goal_map[pos[1], pos[0]] = 1
                goal_pose = [curr_o]
            else:
                inst_map_idx = instance_map == goal_inst
                inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                goal_map = (instance_map[inst_map_idx] == goal_inst).to(torch.float)
        return goal_map, found_goal, goal_pose

    def select_and_localize_instance(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        lmb: torch.Tensor,  # local map boundaries
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
        goal_pose = None
        instance_map = local_map[0][
            MC.NON_SEM_CHANNELS
            + self.num_sem_categories : MC.NON_SEM_CHANNELS
            + 2 * self.num_sem_categories,
            :,
            :,
        ]

        if goal_inst is not None and instance_goal_found is True:
            goal_map, found_goal, goal_pose = self.get_goal_map_from_goal_instance(
                instance_map, goal_map, lmb, goal_inst, instance_goal_found, found_goal
            )
            return goal_map, found_goal, goal_pose, instance_goal_found, goal_inst

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
            goal_map, found_goal, goal_pose = self.get_goal_map_from_goal_instance(
                instance_map, goal_map, lmb, goal_inst, instance_goal_found, found_goal
            )

        return goal_map, found_goal, goal_pose, instance_goal_found, goal_inst
