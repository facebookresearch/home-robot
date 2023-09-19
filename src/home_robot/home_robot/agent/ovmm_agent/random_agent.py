# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import numpy as np

from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)


class RandomAgent(Agent):
    """A random agent that takes random discrete or continuous actions."""

    def __init__(self, config, device_id: int = 0):
        super().__init__()
        self.config = config
        self.snap_probability = 5e-3
        self.desnap_probability = 5e-3
        self.stop_probability = 0.01
        self.max_forward = (
            config.habitat.task.actions.base_velocity.max_displacement_along_axis
        )
        self.max_turn_degrees = (
            config.habitat.task.actions.base_velocity.max_turn_degrees
        )
        self.max_turn_radians = self.max_turn_degrees / 180 * np.pi
        self.max_joints_delta = config.habitat.task.actions.arm_action.max_delta_pos
        self.discrete_actions = config.AGENT.PLANNER.discrete_actions
        self.timestep = 0
        assert (
            self.snap_probability + self.desnap_probability + self.stop_probability
            <= 1.0
        )

    def reset(self):
        """Initialize agent state."""
        self.timestep = 0

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timestep = 0

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timestep = 0

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """Take a random action."""
        action = None
        r = np.random.rand()
        info = {"timestep": self.timestep, "semantic_frame": obs.rgb}
        if r < self.snap_probability:
            action = DiscreteNavigationAction.SNAP_OBJECT
        elif r < self.snap_probability + self.desnap_probability:
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif (
            r < self.snap_probability + self.desnap_probability + self.stop_probability
        ):
            action = DiscreteNavigationAction.STOP
        elif self.discrete_actions:
            action = np.random.choice(
                [
                    DiscreteNavigationAction.MOVE_FORWARD,
                    DiscreteNavigationAction.TURN_LEFT,
                    DiscreteNavigationAction.TURN_RIGHT,
                    DiscreteNavigationAction.EXTEND_ARM,
                    DiscreteNavigationAction.NAVIGATION_MODE,
                    DiscreteNavigationAction.MANIPULATION_MODE,
                ]
            )
        else:
            xyt = np.random.uniform(
                [-self.max_forward, -self.max_forward, -self.max_turn_radians],
                [self.max_forward, self.max_forward, self.max_turn_radians],
            )
            joints = np.random.uniform(
                -self.max_joints_delta, self.max_joints_delta, size=(10,)
            )
            action = ContinuousFullBodyAction(joints, xyt)
        self.timestep += 1
        return action, info, obs
