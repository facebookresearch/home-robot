from abc import abstractmethod
from typing import Any, Dict

import habitat

import home_robot
import home_robot.abstract_env


class HabitatEnv(home_robot.abstract_env.Env):
    def __init__(self, habitat_env: habitat.core.env.Env):
        self.habitat_env = habitat_env
        self._last_obs = None

    def reset(self):
        self._last_obs = self._preprocess_obs(self.habitat_env.reset())

    def apply_action(self, action: home_robot.core_interfaces.Action):
        self._last_obs = self.habitat_env.step(self._preprocess_action(action))

    def get_observation(self) -> home_robot.core_interfaces.Observations:
        return self._last_obs

    @property
    def episode_over(self) -> bool:
        return self.habitat_env.episode_over

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass

    @abstractmethod
    def _preprocess_obs(self,
                        obs: habitat.core.simulator.Observations
                        ) -> home_robot.core_interfaces.Observations:
        pass

    @abstractmethod
    def _preprocess_action(self,
                           action: home_robot.core_interfaces.Action
                           ) -> Any:
        pass
