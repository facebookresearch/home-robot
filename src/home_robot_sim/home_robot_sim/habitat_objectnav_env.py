from typing import Dict, Any

import habitat

from .habitat_abstract_env import HabitatEnv
import home_robot


class HabitatObjectNavEnv(HabitatEnv):
    def __init__(self, habitat_env: habitat.core.env.Env):
        super().__init__(habitat_env)

    def get_episode_metrics(self) -> Dict:
        # TODO
        pass

    def _preprocess_obs(self,
                        obs: habitat.core.simulator.Observations
                        ) -> home_robot.core_interfaces.Observations:
        # TODO
        pass

    def _preprocess_action(self,
                           action: home_robot.core_interfaces.Action
                           ) -> Any:
        # TODO
        pass
