import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class GoatAgentModule(nn.Module):
    def __init__(self, config, instance_memory: Optional[InstanceMemory] = None):
        super().__init__()
        self.instance_memory = instance_memory
        self.goal_inst = None
        self.img_instance_goal_found = False
        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
        )
        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy
        )
        self.goal_policy_config = config.AGENT.SUPERGLUE

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def superglue(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
        all_matches: List = None,
        all_confidences: List = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Goal detection and localization via SuperGlue"""

        score_func = self.goal_policy_config.score_function
        assert score_func in ["confidence_sum", "match_count"]

        if all_matches is not None:
            self.img_instance_goal_found = True
            self.goal_inst = None
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

                if max(max_scores) > self.goal_policy_config.score_thresh:
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
                    goal_map = (instance_map[inst_map_idx] == inst_idx + 1).to(
                        torch.float
                    )

                    if goal_map.any():
                        self.img_instance_goal_found = True
                        self.goal_inst = inst_idx + 1
                        print(f"{self.goal_inst} will be the goal")
                    else:
                        print("Instance was seen, but not present in local map.")
                else:
                    print("Goal image does not match any instance.")
                    # TODO: dont stop at the first instance, but rather find the best one

        if self.goal_inst is not None and self.img_instance_goal_found is True:
            found_goal[0] = True

            instance_map = local_map[0][
                MC.NON_SEM_CHANNELS
                + self.num_sem_categories : MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
                :,
                :,
            ]  # TODO: currently assuming img goal instance was an object outside of the vocabulary
            inst_map_idx = instance_map == self.goal_inst
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
            goal_map = (instance_map[inst_map_idx] == self.goal_inst).to(torch.float)

            # goal_map = (local_map[0][22:38][0] == self.goal_inst).to(torch.float)
        else:
            for e in range(confidence.shape[0]):
                # if the goal category is empty, the goal can't be found
                if not local_map[e, -1].any().item():
                    continue

                if score_func == "confidence_sum":
                    score = confidence[e][matches[e] != -1].sum()
                else:  # match_count
                    score = (matches[e] != -1).sum()

                if score < self.goal_policy_config.score_thresh:
                    continue

                found_goal[e] = True
                # Set goal_map to the last channel of the local semantic map
                goal_map[e, 0] = local_map[e, -1]

        return goal_map, found_goal

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        reject_visited_targets=False,
        blacklist_target=False,
        matches=None,
        confidence=None,
        all_matches=None,
        all_confidences=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation, instance_segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)
            seq_object_goal_category: sequence of object goal categories of shape
             (batch_size, sequence_length, 1)
        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Reset the last channel of the local map each step when found_goal=False
        if matches is not None or confidence is not None:
            init_local_map[:, -1][seq_found_goal[:, 0] == 0] *= 0.0

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
            blacklist_target,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        map_features = seq_map_features.flatten(0, 1)
        # Compute the frontier map here
        frontier_map = self.policy.get_frontier_map(map_features)

        if matches is not None or confidence is not None:
            seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
                seq_found_goal[:, 0] == 0
            ]

            if len(all_matches) > 0:
                seq_goal_map, seq_found_goal = self.superglue(
                    seq_goal_map,
                    seq_found_goal,
                    final_local_map,
                    matches,
                    confidence,
                    all_matches,
                    all_confidences,
                )

            # predict if the goal is found and where it is.
            seq_goal_map, seq_found_goal = self.superglue(
                seq_goal_map, seq_found_goal, final_local_map, matches, confidence
            )
            seq_goal_map = seq_goal_map.view(
                batch_size, sequence_length, *seq_goal_map.shape[-2:]
            )

        else:
            # Predict high-level goals from map features
            # batched across sequence length x num environments
            if seq_object_goal_category is not None:
                seq_object_goal_category = seq_object_goal_category.flatten(0, 1)

            # Compute the goal map
            goal_map, found_goal = self.policy(
                map_features,
                seq_object_goal_category,
                reject_visited_targets=reject_visited_targets,
            )

            seq_goal_map = goal_map.view(
                batch_size, sequence_length, *goal_map.shape[-2:]
            )
            seq_found_goal = found_goal.view(batch_size, sequence_length)

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )