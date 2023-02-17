from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import habitat
from habitat.core.simulator import Observations


class HabitatAgent(habitat.Agent, ABC):
    def __init__(self, home_robot_agent: home_robot.agent.abstract_agent.Agent):
        self.home_robot_agent = home_robot_agent

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, obs: Observations) -> Union[int, str, Dict[str, Any]]:
        pass
