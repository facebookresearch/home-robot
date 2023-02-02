from abc import ABC, abstractmethod

from home_robot.core_interfaces import Action, Observations


class Agent(ABC):
    @abstractmethod
    def reset(self) -> Observations:
        pass

    @abstractmethod
    def act(self, obs: Observations) -> Action:
        pass
