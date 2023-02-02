import habitat

from .habitat_abstract_env import HabitatEnv
from home_robot.core_interfaces import Action, Observations


class HabitatObjectNavEnv(HabitatEnv):
    def __init__(self, habitat_env: habitat.core.env.Env):
        super().__init__(habitat_env)

    def apply_action(self, action: Action):
        # TODO
        pass

    def get_observation(self, action: Observations):
        # TODO
        pass
