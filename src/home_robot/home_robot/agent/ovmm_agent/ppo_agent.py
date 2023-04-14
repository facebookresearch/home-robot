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
from habitat.utils.gym_adapter import (
    continuous_vector_action_to_hab_dict,
    create_action_space,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import batch_obs, get_num_actions

from home_robot.core.interfaces import Observations
from home_robot_sim.env.habitat_objectnav_env.constants import (
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
        self.device_id = device_id
        self.device = (
            torch.device(f"cuda:{self.device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.config = config

        self.hidden_size = config.habitat_baselines.rl.ppo.hidden_size
        random_generator.seed(config.habitat.seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        # TODO: per-skill policy config (currently config parameters are same across skills so shouldn't matter)
        policy = baseline_registry.get_policy(config.habitat_baselines.rl.policy.name)

        # whether the skill uses continuous and discrete actions
        self.continuous_actions = (
            True
            if config.habitat_baselines.rl.policy.action_distribution_type
            != "categorical"
            else False
        )

        # the complete action space
        if self.continuous_actions:
            self.env_action_space = self.action_space
        else:
            self.env_action_space = spaces.Discrete(get_num_actions(self.action_space))

        # read transforms from config
        self.obs_transforms = get_active_obs_transforms(config)

        # obs keys to be passed to the policy. TODO: Read from skill config
        self.skill_obs_keys = [
            "robot_head_depth",
            "object_embedding",
            "object_segmentation",
            "joint",
            "is_holding",
            "relative_resting_position",
        ]
        skill_obs_spaces = spaces.Dict(
            {k: obs_spaces[0].spaces[k] for k in self.skill_obs_keys}
        )

        # actions the skill takes. TODO: Read from skill config
        self.skill_actions = ["arm_action", "base_velocity"]

        # filter the action space, deepcopy is necessary because we override arm_action next
        self.filtered_action_space = spaces.Dict(
            {a: copy.deepcopy(self.action_space[0][a]) for a in self.skill_actions}
        )
        # TODO: read a mask from config that specifies controllable joints
        self.filtered_action_space["arm_action"]["arm_action"] = spaces.Box(
            shape=[0], low=-1.0, high=1.0, dtype=np.float32
        )

        self.vector_action_space = create_action_space(self.filtered_action_space)
        self.num_actions = self.vector_action_space.shape[0]

        # TODO: use skill specific policy config
        self.actor_critic = policy.from_config(
            config,
            skill_obs_spaces,
            self.vector_action_space,
        )

        self.actor_critic.to(self.device)

        # load checkpoint
        model_path = skill_config.CHECKPOINT_PATH
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

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

        self.prev_actions = torch.zeros(
            1,
            self.num_actions,
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
        # TODO: override for GazeAgent or convert all observations to hab observation space here
        return OrderedDict(
            {
                "robot_head_depth": np.expand_dims(normalized_depth, -1),
                "object_embedding": obs.task_observations["object_embedding"],
                "object_segmentation": np.expand_dims(
                    obs.semantic == obs.task_observations["object_goal"], -1
                ).astype(np.uint8),
                "joint": obs.joint,
                "relative_resting_position": obs.relative_resting_position,
                "is_holding": obs.is_holding,
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
                values,
                actions,
                actions_log_probs,
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
            else:
                # TODO: to be tested (by Nav skill?)
                step_action = map_discrete_habitat_actions(
                    action_data.env_actions[0].item()
                )
        # TODO: Read a mask fro controllable arm joints from configs. Set the remaining to zero
        step_action["action_args"]["arm_action"] = np.array([0.0] * 7)
        vis_inputs = {
            "semantic_frame": observations.task_observations["semantic_frame"],
            "goal_name": observations.task_observations["goal_name"],
            "third_person_image": observations.third_person_image,
        }
        return (
            step_action["action_args"],
            vis_inputs,
            self.does_want_terminate(observations, actions),
        )


def map_discrete_habitat_actions(discrete_action):
    # TODO map habitat actions to home-robot actions
    raise NotImplementedError
