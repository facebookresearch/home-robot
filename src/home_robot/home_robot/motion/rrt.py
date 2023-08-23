# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on Caelan Garrett's code from here: https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py


import time
from random import random
from typing import Callable, List

import numpy as np

from home_robot.motion.base import Planner, PlanResult
from home_robot.motion.space import ConfigurationSpace


class TreeNode:
    """Placeholder class"""

    pass


class TreeNode(object):
    """Stores an individual spot in the tree"""

    def __init__(self, state: np.ndarray, parent=None):
        """A treenode is just a pointer back to its parent and an associated state."""
        self.state = state
        self.parent = parent

    def backup(self) -> List[TreeNode]:
        """Get the full plan by looking back from this point. Returns a list of TreeNodes which contain state."""
        sequence = []
        node = self
        # Look backwards to get a tree
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]


class RRT(Planner):
    """Define RRT planning problem and parameters"""

    def __init__(
        self,
        space: ConfigurationSpace,
        validate_fn: Callable,
        max_iterations=1000,
        max_runtime=None,
    ):
        """Create RRT planner with configuration"""
        super(RRT, self).__init__(space, validate_fn)

    def plan(self, q0, qg) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py
        """
        # start_time = time.time()
        if self.validate(q0):
            return PlanResult(False)

        raise NotImplementedError("RRT not finished")
