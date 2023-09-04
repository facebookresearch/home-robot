import os
from typing import Union

import numpy as np
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from home_robot.core.interfaces import DiscreteNavigationAction

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self):
        self.env = None
        self.shortest_path_follower = None
        self.goal_coordinates = None  # needs to be a list, and the agent is implemented to follow one after the other in order
        self.discrete_action_map = {
            HabitatSimActions.stop: DiscreteNavigationAction.STOP,
            HabitatSimActions.move_forward: DiscreteNavigationAction.MOVE_FORWARD,
            HabitatSimActions.turn_left: DiscreteNavigationAction.TURN_LEFT,
            HabitatSimActions.turn_right: DiscreteNavigationAction.TURN_RIGHT,
        }
        self.current_goal = (
            0  # index of the current goal in the list of goal coordinates
        )
        self.coarse_navigation = False

    def set_oracle_info(self, env, goal_coordinates, goal_radius=0.5):
        """Instantiate shortest path follower

        Args:
            env: Habitat env
            goal_coordinates: List of xyz goal coordinates. Agent implemented to follow one after the other in order
        """
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=env.habitat_env.sim,
            goal_radius=goal_radius,
            return_one_hot=False,
        )
        self.goal_coordinates = goal_coordinates
        self.current_goal = (
            0  # index of the current goal in the list of goal coordinates
        )

    def act(self, observations, info) -> Union[int, np.ndarray]:
        action = self.discrete_action_map[
            self.shortest_path_follower.get_next_action(
                self.goal_coordinates[self.current_goal]
            )
        ]
        print(f"Oracle action: {action}")
        print(f"Goal: {self.goal_coordinates[self.current_goal]}")
        print(f"Agent: {self.shortest_path_follower._sim.robot.base_pos}")

        terminate = False
        if action == DiscreteNavigationAction.STOP:
            if self.current_goal >= len(self.goal_coordinates) - 1:
                terminate = True  # completed all goals
            else:
                print("Reached goal! Moving to next goal...")
                print(f"Curr goal: {self.goal_coordinates[self.current_goal]}")
                print(f"Next goal: {self.goal_coordinates[self.current_goal+1]}")
                self.current_goal += 1  # move to next goal
                return self.act(observations, info)

        return action, terminate

    def reset(self) -> None:
        self.env = None
        self.shortest_path_follower = None
        self.goal_coordinates = None
        self.goal_candidate = 0

    def reset_vectorized(self) -> None:
        self.reset()  # or NotImplementedError, really.
