from abc import abstractmethod
from typing import Any, Dict, Optional

import home_robot.core.abstract_env
from home_robot.core.interfaces import Action, Observations


class SpotEnv(home_robot.core.abstract_env):
    def __init__(self):
        self.spot_env = 2  # TODO

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        pass

    @abstractmethod
    def get_observation(self) -> Observations:
        pass
        # observations = self.spot_env.get_observations()

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass
