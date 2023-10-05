# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Node(ABC):
    """Placeholder containing just a state."""

    def __init__(self, state):
        self.state = state


class ConfigurationSpace(ABC):
    """class defining a region over which we can sample parameters"""

    def __init__(self, dof: int, mins, maxs, step_size: float = 0.1):
        self.dof = dof
        self.step_size = step_size
        self.update_bounds(mins, maxs)
        assert self.step_size > 0.0

    def update_bounds(self, mins, maxs):
        assert len(mins) == self.dof, "mins' length must be equal to the space dof"
        assert len(maxs) == self.dof, "maxs' length must be equal to space dof"
        self.mins = mins
        self.maxs = maxs
        self.ranges = maxs - mins

    def sample(self) -> np.ndarray:
        return (np.random.random(self.dof) * self.ranges) + self.mins

    def distance(self, q0, q1) -> float:
        """Return distance between q0 and q1."""
        return np.linalg.norm(q0 - q1)

    def extend(self, q0, q1):
        """extend towards another configuration in this space"""
        dq = q1 - q0
        step = dq / np.linalg.norm(dq) * self.step_size
        if self.distance(q0, q1) > self.step_size:
            qi = q0 + step
            while self.distance(qi, q1) > self.step_size:
                qi = qi + step
                yield qi
        yield q1

    def closest_node_to_state(self, state, nodes: List[Node]):
        """returns closest node to a given state"""
        min_dist = float("Inf")
        min_node = None
        for node in nodes:
            dist = self.distance(node.state, state)
            if dist < min_dist:
                min_dist = dist
                min_node = node
        return min_node


class XYT(ConfigurationSpace):
    """Space for (x, y, theta) base rotations"""

    def __init__(self, mins: np.ndarray = None, maxs: np.ndarray = None):
        """Create XYT space with some defaults"""
        if mins is None:
            mins = np.array([-10, -10, -np.pi])
        if maxs is None:
            maxs = np.array([10, 10, np.pi])
        super(XYT, self).__init__(3, mins, maxs)

    def update_bounds(self, mins, maxs):
        """Update bounds for just x and y sometimes, since that's all that will be changing"""
        if len(mins) == 3:
            super().update_bounds(mins, maxs)
        elif len(mins) == 2:
            assert len(mins) == len(maxs), "min and max bounds must match"
            # Just update x and y
            self.mins[:2] = mins
            self.maxs[:2] = maxs
            self.ranges[:2] = maxs - mins
