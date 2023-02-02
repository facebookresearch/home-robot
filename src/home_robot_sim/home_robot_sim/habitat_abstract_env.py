from abc import abstractmethod

import habitat

import home_robot
from home_robot.core_interfaces import Action, Observations


class HabitatEnv(home_robot.abstract_env.Env):
    def __init__(self, habitat_env: habitat.core.env.Env):
        self.habitat_env = habitat_env

    @abstractmethod
    def apply_action(self, action: Action):
        pass

    @abstractmethod
    def get_observation(self, action: Observations):
        pass
