#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import random
from collections import OrderedDict
from typing import Any, Tuple, Union

import gym.spaces as spaces
import numba
import numpy as np
import torch
from habitat.core.agent import Agent
from habitat.core.spaces import EmptySpace
from habitat.gym.gym_wrapper import (
    continuous_vector_action_to_hab_dict,
    create_action_space,
)
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat_baselines.utils.common import batch_obs
from omegaconf import OmegaConf

import home_robot.utils.pose as pu
from home_robot.agent.ovmm_agent.complete_obs_space import get_complete_obs_space
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)

random_generator = np.random.RandomState()


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _seed_numba(seed)
    torch.random.manual_seed(seed)


def sample_random_seed():
    set_random_seed(random_generator.randint(2**32))


class PPOAgent(Agent):
    """
    Abstract class for evaluation of a PPO policy/skill. Loads the trained skill and takes actions
    """

    def __init__(
        self,
        config,
        skill_config,
        device_id: int = 0,
        obs_spaces=None,
        action_spaces=None,
    ) -> None:
        # Observation and action spaces for the full task
        self.obs_space = obs_spaces
        self.action_space = action_spaces
        if obs_spaces is None:
            self.obs_space = [
                get_complete_obs_space(skill_config, config),
            ]
        if action_spaces is None:
            self.action_space = [
                spaces.dict.Dict(
                    {
                        "arm_action": spaces.dict.Dict(
                            {
                                "arm_action": spaces.Box(-1.0, 1.0, (7,), np.float32),
                                "grip_action": spaces.Box(-1.0, 1.0, (1,), np.float32),
                            }
                        ),
                        "base_velocity": spaces.dict.Dict(
                            {
                                "base_vel": spaces.Box(-20.0, 20.0, (3,), np.float32),
                            }
                        ),
                        "rearrange_stop": EmptySpace(),
                        "manipulation_mode": EmptySpace(),
                    }
                )
            ]
        self.device_id = device_id
        self.device = (
            torch.device(f"cuda:{self.device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.config = config
        # Read in the RL config (in hydra format)
        self.rl_config = get_habitat_config(skill_config.rl_config)
        self.hidden_size = self.rl_config.habitat_baselines.rl.ppo.hidden_size
        random_generator.seed(config.seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        policy = baseline_registry.get_policy(
            self.rl_config.habitat_baselines.rl.policy.name
        )

        # whether the skill uses continuous and discrete actions
        self.continuous_actions = (
            self.rl_config.habitat_baselines.rl.policy.action_distribution_type
            != "categorical"
        )

        # read transforms from config
        self.obs_transforms = get_active_obs_transforms(self.rl_config)

        # obs keys to be passed to the policy
        self.skill_obs_keys = skill_config.gym_obs_keys
        skill_obs_spaces = spaces.Dict(
            {k: self.obs_space[0].spaces[k] for k in self.skill_obs_keys}
        )

        # actions the skill takes
        self.skill_actions = skill_config.allowed_actions
        if self.continuous_actions:
            # filter the action space, deepcopy is necessary because we override arm_action next
            self.filtered_action_space = spaces.Dict(
                {a: copy.deepcopy(self.action_space[0][a]) for a in self.skill_actions}
            )
        else:
            self.filtered_action_space = spaces.Dict(
                {a: EmptySpace() for a in self.skill_actions}
            )

        # The policy may not control all arm joints, read the mask that indicates the joints controlled by the policy
        if (
            "arm_action" in self.filtered_action_space.spaces
            and "arm_action" in self.filtered_action_space["arm_action"].spaces
        ):
            self.arm_joint_mask = skill_config.arm_joint_mask
            self.num_arm_joints_controlled = np.sum(skill_config.arm_joint_mask)
            self.filtered_action_space["arm_action"]["arm_action"] = spaces.Box(
                shape=[self.num_arm_joints_controlled],
                low=-1.0,
                high=1.0,
                dtype=np.float32,
            )

        self.vector_action_space = create_action_space(self.filtered_action_space)
        if self.continuous_actions:
            self.num_actions = self.vector_action_space.shape[0]
            self.actions_dim = self.num_actions
        else:
            self.num_actions = self.vector_action_space.n
            self.actions_dim = 1

        # load checkpoint
        model_path, ckpt = skill_config.checkpoint_path, None
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)

        # set `normalize_visual_inputs` in config if the checkpoint has running mean and var parameters in the visual encoder
        OmegaConf.set_readonly(self.rl_config, False)
        self.rl_config.habitat_baselines.rl.ddppo.normalize_visual_inputs = (
            "net.visual_encoder.running_mean_and_var._mean" in ckpt["state_dict"].keys()
        )

        # Initialize actor critic using the policy config
        self.actor_critic = policy.from_config(
            self.rl_config,
            skill_obs_spaces,
            self.vector_action_space,
        )
        self.actor_critic.to(self.device)
        if ckpt:
            self.actor_critic.load_state_dict(ckpt["state_dict"])
        self.actor_critic.eval()

        self.max_displacement = skill_config.max_displacement
        self.max_turn = skill_config.max_turn_degrees * np.pi / 180
        self.min_displacement = skill_config.min_displacement
        self.min_turn = skill_config.min_turn_degrees * np.pi / 180
        self.nav_goal_seg_channels = skill_config.nav_goal_seg_channels
        self.max_joint_delta = None
        self.min_joint_delta = None
        if "arm_action" in skill_config.allowed_actions:
            self.max_joint_delta = skill_config.max_joint_delta
            self.grip_threshold = skill_config.grip_threshold
            self.min_joint_delta = skill_config.min_joint_delta
        if "manipulation_mode" in skill_config.allowed_actions:
            self.manip_mode_threshold = skill_config.manip_mode_threshold
            self.constraint_base_in_manip_mode = (
                skill_config.constraint_base_in_manip_mode
            )
        self.discrete_forward = None
        self.discrete_turn = None
        if "move_forward" in skill_config.allowed_actions:
            self.discrete_forward = skill_config.discrete_forward
        if (
            "turn_left" in skill_config.allowed_actions
            or "turn_right" in skill_config.allowed_actions
        ):
            self.discrete_turn = skill_config.discrete_turn_degrees * np.pi / 180
        self.terminate_condition = skill_config.terminate_condition
        self.show_rl_obs = getattr(config, "SHOW_RL_OBS", False)
        self.manip_mode_called = False
        self.skill_start_gps = None
        self.skill_start_compass = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
            dtype=torch.float32,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

        self.prev_actions = torch.zeros(
            1,
            self.actions_dim,
            dtype=torch.float32 if self.continuous_actions else torch.long,
            device=self.device,
        )
        self.manip_mode_called = False
        self.skill_start_gps = None
        self.skill_start_compass = None

    def reset_vectorized(self):
        """Initialize agent state."""
        self.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state."""
        self.reset()

    def does_want_terminate(self, observations, action) -> bool:
        if not self.continuous_actions and self.terminate_condition == "discrete_stop":
            return action == DiscreteNavigationAction.STOP
        elif self.terminate_condition == "grip":
            return action["grip_action"][0] >= self.grip_threshold
        elif self.terminate_condition == "ungrip":
            return action["grip_action"][0] < self.grip_threshold
        elif self.terminate_condition == "obj_goal_at_center":
            # check if the center pixel corresponds to the object of interest
            h, w = observations.semantic.shape
            return (
                observations.semantic[h // 2, w // 2]
                == observations.task_observations["object_goal"]
            )
        elif (
            "rearrange_stop" in action and self.terminate_condition == "continuous_stop"
        ):
            return action["rearrange_stop"][0] > 0
        else:
            raise ValueError("Invalid terminate condition")

    def _get_goal_segmentation(self, obs: Observations) -> np.ndarray:
        if self.nav_goal_seg_channels == 1:
            return np.expand_dims(
                obs.semantic == obs.task_observations["end_recep_goal"], -1
            ).astype(np.uint8)
        elif self.nav_goal_seg_channels == 2:
            object_goal = np.expand_dims(
                obs.semantic == obs.task_observations["object_goal"], -1
            ).astype(np.uint8)
            start_recep_goal = np.expand_dims(
                obs.semantic == obs.task_observations["start_recep_goal"], -1
            ).astype(np.uint8)
            return np.concatenate([object_goal, start_recep_goal], axis=-1)

    def _get_receptacle_segmentation(self, obs: Observations) -> np.ndarray:
        rec_seg = obs.semantic
        recep_idx_start = obs.task_observations["recep_idx"]
        max_val = obs.task_observations["semantic_max_val"]
        seg_map = np.zeros(max_val + 1, dtype=np.int32)
        for obj_idx in range(recep_idx_start):
            seg_map[obj_idx] = 0
        for i, obj_idx in enumerate(range(recep_idx_start, max_val)):
            seg_map[obj_idx] = i + 1
        seg_map[max_val] = 0
        rec_seg = seg_map[rec_seg]
        return rec_seg[..., np.newaxis].astype(np.int32)

    def convert_to_habitat_obs_space(
        self, obs: Observations
    ) -> "OrderedDict[str, Any]":
        # normalize depth
        min_depth = self.config.ENVIRONMENT.min_depth
        max_depth = self.config.ENVIRONMENT.max_depth
        normalized_depth = obs.depth.copy()
        normalized_depth[normalized_depth == MIN_DEPTH_REPLACEMENT_VALUE] = min_depth
        normalized_depth[normalized_depth == MAX_DEPTH_REPLACEMENT_VALUE] = max_depth
        normalized_depth = np.clip(normalized_depth, min_depth, max_depth)
        normalized_depth = (normalized_depth - min_depth) / (max_depth - min_depth)
        rel_pos = pu.get_rel_pose_change(
            [obs.gps[0], obs.gps[1], obs.compass],
            [
                self.skill_start_gps[0],
                self.skill_start_gps[1],
                self.skill_start_compass,
            ],
        )
        hab_obs = OrderedDict(
            {
                "object_category": obs.task_observations["object_category"],
                "head_depth": np.expand_dims(normalized_depth, -1).astype(np.float32),
                "object_embedding": obs.task_observations["object_embedding"],
                "object_segmentation": np.expand_dims(
                    obs.semantic == obs.task_observations["object_goal"], -1
                ).astype(np.uint8),
                "goal_recep_segmentation": np.expand_dims(
                    obs.semantic == obs.task_observations["end_recep_goal"], -1
                ).astype(np.uint8),
                "start_recep_segmentation": np.expand_dims(
                    obs.semantic == obs.task_observations["start_recep_goal"], -1
                ).astype(np.uint8),
                "joint": obs.joint,
                "relative_resting_position": obs.relative_resting_position,
                "is_holding": np.array(obs.task_observations["prev_grasp_success"]),
                "robot_start_gps": np.array((rel_pos[1].item(), rel_pos[0].item())),
                "robot_start_compass": pu.normalize_radians(rel_pos[2]),
                "start_receptacle": np.array(obs.task_observations["start_receptacle"]),
                "goal_receptacle": np.array(obs.task_observations["goal_receptacle"]),
            }
        )

        if "ovmm_nav_goal_segmentation" in self.skill_obs_keys:
            hab_obs["ovmm_nav_goal_segmentation"] = self._get_goal_segmentation(obs)
        if "receptacle_segmentation" in self.skill_obs_keys:
            hab_obs["receptacle_segmentation"] = self._get_receptacle_segmentation(obs)
        return hab_obs

    def act(
        self, observations: Observations, info
    ) -> Tuple[
        Union[
            ContinuousFullBodyAction,
            ContinuousNavigationAction,
            DiscreteNavigationAction,
        ],
        bool,
    ]:
        sample_random_seed()
        if self.skill_start_gps is None:
            self.skill_start_gps = observations.gps
        if self.skill_start_compass is None:
            self.skill_start_compass = observations.compass
        obs = self.convert_to_habitat_obs_space(observations)
        viz_obs = {k: obs[k] for k in self.skill_obs_keys}
        batch = batch_obs([obs], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        for k in self.skill_obs_keys:
            viz_obs[k + "_resized"] = batch[k][0].cpu().numpy()
        batch = OrderedDict([(k, batch[k]) for k in self.skill_obs_keys])

        if self.show_rl_obs:
            frame = observations_to_image(viz_obs, info={})
            info["rl_obs_frame"] = frame

        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )

            actions, self.test_recurrent_hidden_states = (
                action_data.actions,
                action_data.rnn_hidden_states,
            )

            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore
            if self.continuous_actions:
                # Clipping actions to the specified limits
                act = np.clip(
                    actions.cpu().numpy(),
                    self.vector_action_space.low,
                    self.vector_action_space.high,
                )
                step_action = continuous_vector_action_to_hab_dict(
                    self.filtered_action_space, self.vector_action_space, act[0]
                )

                robot_action = self._map_continuous_habitat_actions(
                    step_action["action_args"]
                )
                does_want_terminate = self.does_want_terminate(
                    observations, step_action["action_args"]
                )
                return (
                    robot_action,
                    info,
                    does_want_terminate,
                )
            else:
                step_action = self._map_discrete_habitat_actions(
                    actions.item(), self.skill_actions
                )
                return (
                    step_action,
                    info,
                    self.does_want_terminate(observations, step_action),
                )

    def _map_continuous_habitat_actions(self, cont_action):
        """Map habitat continuous actions to home-robot continuous actions"""
        # TODO: add home-robot support for simultaneous gripping
        if (
            not self.manip_mode_called
            and "manipulation_mode" in cont_action
            and cont_action["manipulation_mode"] >= self.manip_mode_threshold
        ):
            self.manip_mode_called = True
            # Todo, look at the order in which hab-lab updates actions
            return DiscreteNavigationAction.MANIPULATION_MODE
        waypoint, turn, sel = cont_action[
            "base_vel"
        ]  # Teleport action after disabling simultaneous turn and lateral movement
        xyt = [0, 0, 0]
        if sel >= 0:
            absolute_forward_dist = np.clip(waypoint, -1, 1) * self.max_displacement
            if np.abs(absolute_forward_dist) < self.min_displacement:
                absolute_forward_dist = 0
            xyt = np.array([absolute_forward_dist, 0, 0])  # forward
        else:
            absolute_turn = np.clip(turn, -1, 1) * self.max_turn
            if np.abs(absolute_turn) < self.min_turn:
                absolute_turn = 0
            xyt = np.array([0, 0, absolute_turn])  # turn

        if self.manip_mode_called and self.constraint_base_in_manip_mode:
            xyt = np.array([0, 0, 0])

        joints = None
        # Map policy controlled arm_action to complete arm_action space
        if "arm_action" in cont_action:
            complete_arm_action = np.array([0.0] * len(self.arm_joint_mask))
            controlled_joint_indices = np.nonzero(self.arm_joint_mask)
            # map the policy controlled joints to "controllable" action space
            complete_arm_action[controlled_joint_indices] = cont_action["arm_action"]
            # add zeros for arm_1, arm_2, arm_3 extensions
            complete_arm_action = np.concatenate(
                [complete_arm_action[:1], [0] * 3, complete_arm_action[1:]]
            )
            joints = np.clip(complete_arm_action, -1, 1) * self.max_joint_delta
            # remove small joint movements
            joints[np.abs(joints) < self.min_joint_delta] = 0
            return ContinuousFullBodyAction(joints, xyt=xyt)
        else:
            return ContinuousNavigationAction(xyt)

    def _map_discrete_habitat_actions(self, discrete_action, skill_actions):
        discrete_action = skill_actions[discrete_action]
        cont_action = np.array([0.0, 0.0, 0.0])
        if discrete_action == "move_forward":
            cont_action[0] = self.discrete_forward
        elif discrete_action == "turn_left":
            cont_action[2] = self.discrete_turn
        elif discrete_action == "turn_right":
            cont_action[2] = -self.discrete_turn
        elif discrete_action == "snap":
            return DiscreteNavigationAction.SNAP_OBJECT
        elif discrete_action == "stop":
            return DiscreteNavigationAction.STOP
        else:
            raise ValueError
        return ContinuousNavigationAction(cont_action)
