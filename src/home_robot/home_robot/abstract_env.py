from abc import ABC, abstractmethod

from .core_interfaces import Action, Observations


class Env(ABC):
    @abstractmethod
    def apply_action(self, action: Action):
        pass

    @abstractmethod
    def get_observation(self) -> Observations:
        pass
