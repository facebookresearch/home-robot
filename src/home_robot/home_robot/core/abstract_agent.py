from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .interfaces import Action, Observations


class Agent(ABC):
    """
    Base home_robot agent that can interact with a simulator or hardware.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, obs: Observations) -> Tuple[Action, Dict[str, Any]]:
        """
        Act end-to-end.

        Arguments:
            obs: home_robot observation

        Returns:
            action: home_robot action
            info: additional information (e.g., for debugging, visualization)
        """
        pass
