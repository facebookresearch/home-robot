#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import random
from collections import OrderedDict
from typing import Any, Dict

import gym.spaces as spaces
import numba
import numpy as np
import torch
from habitat.core.agent import Agent
from habitat.core.spaces import EmptySpace
from habitat.utils.gym_adapter import (
    continuous_vector_action_to_hab_dict,
    create_action_space,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat_baselines.utils.common import batch_obs

from home_robot.core.interfaces import (
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
                spaces.dict.Dict(
                    {
                        "is_holding": spaces.Box(0.0, 1.0, (1,), np.float32),
                        "robot_head_depth": spaces.Box(
                            0.0, 1.0, (256, 256, 1), np.float32
                        ),
                        "joint": spaces.Box(
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).max,
                            (10,),
                            np.float32,
                        ),
                        "object_embedding": spaces.Box(
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).max,
                            (512,),
                            np.float32,
                        ),
                        "relative_resting_position": spaces.Box(
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).max,
                            (3,),
                            np.float32,
                        ),
                        "object_segmentation": spaces.Box(
                            0.0, 1.0, (256, 256, 1), np.uint8
                        ),
                    }
                )
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
                                "base_vel": spaces.Box(-20.0, 20.0, (2,), np.float32),
                            }
                        ),
                        "extend_arm": EmptySpace(),
                        "face_arm": EmptySpace(),
                        "rearrange_stop": EmptySpace(),
                        "reset_joints": EmptySpace(),
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

        # Initialize actor critic using the policy config
        self.actor_critic = policy.from_config(
            self.rl_config,
            skill_obs_spaces,
            self.vector_action_space,
        )
        self.actor_critic.to(self.device)

        # load checkpoint
        model_path = skill_config.checkpoint_path
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )
        self.actor_critic.eval()
        self.max_forward = skill_config.max_forward
        self.max_turn = skill_config.max_turn

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

    def reset_vectorized(self):
        """Initialize agent state."""
        self.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state."""
        self.reset()

    def does_want_terminate(self, observations, actions):
        # TODO: override in GazeAgent
        # raise NotImplementedError
        # For Gaze check if the center pixel corresponds to the object of interest
        if not self.continuous_actions:
            return actions == DiscreteNavigationAction.STOP
        h, w = observations.semantic.shape
        return (
            observations.semantic[h // 2, w // 2]
            == observations.task_observations["object_goal"]
        )

    def convert_to_habitat_obs_space(
        self, obs: Observations
    ) -> "OrderedDict[str, Any]":

        # normalize depth
        min_depth = self.config.ENVIRONMENT.min_depth
        max_depth = self.config.ENVIRONMENT.max_depth
        normalized_depth = obs.depth.copy()
        normalized_depth[normalized_depth == MIN_DEPTH_REPLACEMENT_VALUE] = 0
        normalized_depth[normalized_depth == MAX_DEPTH_REPLACEMENT_VALUE] = 1
        normalized_depth = (normalized_depth - min_depth) / (max_depth - min_depth)
        # TODO: convert all observations to hab observation space here
        return OrderedDict(
            {
                "robot_head_depth": np.expand_dims(normalized_depth, -1).astype(
                    np.float32
                ),
                "object_embedding": obs.task_observations["object_embedding"],
                "object_segmentation": np.expand_dims(
                    obs.semantic == obs.task_observations["object_goal"], -1
                ).astype(np.uint8),
                "joint": obs.joint,
                "relative_resting_position": obs.relative_resting_position,
                "is_holding": obs.is_holding,
                "robot_start_gps": np.array((obs.gps[0], -1 * obs.gps[1])),
                "robot_start_compass": obs.compass + np.pi / 2,
                "receptacle_segmentation": obs.task_observations[
                    "receptacle_segmentation"
                ],
                "cat_nav_goal_segmentation": obs.task_observations[
                    "cat_nav_goal_segmentation"
                ],
                "start_receptacle": obs.task_observations["start_receptacle"],
            }
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        sample_random_seed()
        obs = self.convert_to_habitat_obs_space(observations)
        batch = batch_obs([obs], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        batch = OrderedDict([(k, batch[k]) for k in self.skill_obs_keys])

        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = action_data

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
                return (
                    self._map_continuous_habitat_actions(step_action["action_args"]),
                    self.does_want_terminate(observations, actions),
                )
            else:
                step_action = self._map_discrete_habitat_actions(
                    actions.item(), self.skill_actions
                )

        if self.continuous_actions:
            # Map policy controlled arm_action to complete arm_action space
            if "arm_action" in step_action["action_args"]:
                complete_arm_action = np.array([0.0] * len(self.arm_joint_mask))
                controlled_joint_indices = np.nonzero(self.arm_joint_mask)
                complete_arm_action[controlled_joint_indices] = step_action[
                    "action_args"
                ]["arm_action"]
                step_action["action_args"]["arm_action"] = complete_arm_action

            return (
                step_action["action_args"],
                self.does_want_terminate(observations, step_action),
            )
        else:
            return (
                step_action,
                self.does_want_terminate(observations, step_action),
            )

    def _map_continuous_habitat_actions(self, cont_action):
        """Map habitat continuous actions to home-robot continuous actions"""
        # TODO: support simultaneous manipulation
        waypoint, sel = cont_action["base_vel"]
        if sel >= 0:
            action = ContinuousNavigationAction(
                [np.clip(waypoint, -1, 1) * self.max_forward, 0, 0]
            )  # forward
        else:
            action = ContinuousNavigationAction(
                [0, 0, np.clip(waypoint, -1, 1) * self.max_turn]
            )  # turn
        return action

    def _map_discrete_habitat_actions(self, discrete_action, skill_actions):
        discrete_action = skill_actions[discrete_action]
        if discrete_action == "move_forward":
            return DiscreteNavigationAction.MOVE_FORWARD
        elif discrete_action == "turn_left":
            return DiscreteNavigationAction.TURN_LEFT
        elif discrete_action == "turn_right":
            return DiscreteNavigationAction.TURN_RIGHT
        elif discrete_action == "stop":
            return DiscreteNavigationAction.STOP
        else:
            raise ValueError
