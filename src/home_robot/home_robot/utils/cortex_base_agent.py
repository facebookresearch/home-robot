from abc import ABC, abstractmethod
from typing import Dict, Type


class BaseAgent(ABC):
    def __init__(self, cfg: "Type", overrides: "Dict" = None):
        """
        Agent constructor

        Args:
            cfg: Omegaconf config object
            env: Env

        """
        self.cfg = cfg
        self.overrides = overrides if overrides is not None else {}

        self.state_memory = {}

        self.initialize()

    @abstractmethod
    def act_on_observations(self, episode_id, obs, goal=None):
        """
        Process environment observation and select action

        Args:
            episode_id: int
            obs: Obs
            goal: string
            new_episode: bool
            episode_complete: bool

        Return values:
            action: string
        """

    @abstractmethod
    def initialize(self):
        pass

    def episode_complete(self, episode_id):
        pass

    # @abstractmethod
    # def reset(self):
    #     """
    #     Reset agent for the beginning of a new episode. I.e. delete
    #     any stored episode memory or running state that the agent uses
    #     to decide actions

    #     Args:
    #         None

    #     Return values:
    #         None

    #     """
    #     pass


# class InteractiveTaskAgent(TaskAgent):

#     def run(self, env):


# class SingleStepRPCTaskAgent(TaskAgent):

#     def run(self):
