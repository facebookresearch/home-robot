import abc

import torch

import home_robot

class Agent(abc.ABC):
    def __init__(self, perception: home_robot.Perception, control: home_robot.Control):
        self.perception = perception

    def reset(self, env):
        pass

    def act(self, obs: home_robot.Observation) -> home_robot.Action:
        perception_output = self.perception.forward(obs)

        # agent logic here

        return self.control(...)