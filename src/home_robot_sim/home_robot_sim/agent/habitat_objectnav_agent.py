from typing import Any, Union, Dict

from habitat.core.simulator import Observations

from .habitat_abstract_agent import HabitatAgent


class HabitatObjectNavAgent(HabitatAgent):
    def __init__(self, home_robot_agent: home_robot.agent.objectnav_agent.ObjectNavAgent):
        super().__init__(home_robot_agent)

    def reset(self):
        # TODO
        pass

    def act(self, obs: Observations) -> Union[int, str, Dict[str, Any]]:
        # TODO
        pass
