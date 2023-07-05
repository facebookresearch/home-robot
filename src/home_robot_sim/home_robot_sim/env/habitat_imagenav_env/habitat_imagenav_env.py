# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, cast

import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig

import home_robot
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv


class HabitatImageNavEnv(HabitatEnv):
    """
    This class is a Habitat simulation environment for performing an Image Goal
    Navigation task (InstanceImageNav) using Home Robot APIs.
    Task definition: https://arxiv.org/pdf/2211.15876
    """

    def __init__(self, habitat_env: habitat.core.env.Env, config: DictConfig) -> None:
        super().__init__(habitat_env)

        self.min_depth = (
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        )
        self.max_depth = (
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        )

    def reset(self) -> None:
        """Initialize a new episode."""
        habitat_obs = self.habitat_env.reset()
        self._last_obs = self._preprocess_obs(habitat_obs)

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        """Translate Habitat observations into home_robot observations."""
        depth = self._preprocess_depth(habitat_obs["depth"])

        task_observations = {"instance_imagegoal": habitat_obs["instance_imagegoal"]}
        metrics = self.get_episode_metrics()
        for k in ["collisions", "top_down_map"]:
            if k in metrics:
                task_observations[k] = metrics[k]

        gps = habitat_obs["gps"]
        gps[1] = -gps[1]
        return home_robot.core.interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=gps,
            task_observations=task_observations,
        )

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_action(self, action: home_robot.core.interfaces.Action) -> int:
        """Translate a home_robot action into a Habitat action."""
        discrete_action = cast(
            home_robot.core.interfaces.DiscreteNavigationAction, action
        )
        return HabitatSimActions[discrete_action.name.lower()]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        """Process info given along with the action."""
        pass
