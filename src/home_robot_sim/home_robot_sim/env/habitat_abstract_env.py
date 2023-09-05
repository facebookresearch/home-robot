# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import abstractmethod
from typing import Any, Dict, Optional, TypeVar

import habitat
import numpy as np
from gym import spaces

import home_robot
import home_robot.core.abstract_env

ActType = TypeVar("ActType")


class HabitatEnv(home_robot.core.abstract_env.Env):
    """
    Base environment that lets a home_robot agent interact with the Habitat
    simulator.

    Subclasses need to implement translation from Habitat observations into
    home_robot observations and from home_robot actions back into Habitat
    actions.
    """

    def __init__(self, habitat_env: habitat.core.env.Env):
        self.habitat_env = habitat_env
        self._last_obs: Optional[habitat.core.interfaces.Observations] = None

    def reset(self):
        self._last_obs = self._preprocess_obs(self.habitat_env.reset())
        return self._last_obs

    def apply_action(
        self,
        action: home_robot.core.interfaces.Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[home_robot.core.interfaces.Observations] = None,
    ):
        if info is not None:
            self._process_info(info)
        habitat_action = self._preprocess_action(action)
        habitat_obs = self.habitat_env.step(habitat_action)
        self._last_obs = self._preprocess_obs(habitat_obs)

    def get_observation(self) -> home_robot.core.interfaces.Observations:
        return self._last_obs

    @property
    def episode_over(self) -> bool:
        return self.habitat_env.episode_over

    def get_episode_metrics(self) -> Dict:
        return self.habitat_env.get_metrics()

    @abstractmethod
    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        """Translate Habitat observations into home_robot observations."""
        pass

    def _preprocess_xy(self, xy: np.array) -> np.array:
        """Translate Habitat navigation (x, y) (i.e., GPS sensor) into robot (x, y)."""
        return np.array([xy[0], -1 * xy[1]])

    @abstractmethod
    def _preprocess_action(
        self,
        action: home_robot.core.interfaces.Action,
    ) -> Any:
        """Translate a home_robot action into a Habitat action."""
        pass

    @abstractmethod
    def _process_info(self, info: Dict[str, Any]) -> Any:
        """Process info given along with the action."""
        pass

    @property
    def observation_space(self):
        return self.habitat_env.observation_space

    def close(self):
        return self.habitat_env.close()

    def seed(self, seed=None):
        return self.habitat_env.seed(seed)

    @property
    def action_space(self) -> spaces.Space[ActType]:
        return self.habitat_env.action_space

    def current_episode(self, all_info: bool = False) -> int:
        return self.habitat_env.current_episode(all_info)

    @property
    def number_of_episodes(self) -> int:
        return self.habitat_env.number_of_episodes

    @property
    def original_action_space(self) -> spaces.space:
        return self.habitat_env.original_action_space
