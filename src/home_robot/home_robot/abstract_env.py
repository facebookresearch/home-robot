from typing import Dict
from abc import ABC, abstractmethod

from .core_interfaces import Action, Observations


class Env(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_action(self, action: Action):
        pass

    @abstractmethod
    def get_observation(self) -> Observations:
        pass

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass
