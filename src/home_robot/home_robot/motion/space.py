# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod

import numpy as np


class ConfigurationSpace(ABC):
    """class defining a region over which we can sample parameters"""

    def __init__(self, dof: int, mins, maxs):
        self.dof = dof
        self.update_bounds(mins, maxs)

    def update_bounds(self, mins, maxs):
        assert len(mins) == self.dof, "mins' length must be equal to the space dof"
        assert len(maxs) == self.dof, "maxs must be equal to space dof"
        self.mins = mins
        self.maxs = maxs
        self.rngs = maxs - mins

    def sample_uniform(self) -> np.ndarray:
        return (np.random.random(self.dof) * self.rngs) + self.mins

    @abstractmethod
    def distance(self, q0, q1) -> float:
        """Return distance between q0 and q1"""
        raise NotImplementedError()

    @abstractmethod
    def extend(self, q0, q1, step_size=0.1):
        """extend towards another configuration in this space"""
        raise NotImplementedError()


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
            self.rngs[:2] = maxs - mins
