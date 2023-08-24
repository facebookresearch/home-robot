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
from home_robot.motion.space import ConfigurationSpace, Node


class TreeNode:
    """Placeholder class"""

    pass


class TreeNode(Node):
    """Stores an individual spot in the tree"""

    def __init__(self, state: np.ndarray, parent=None):
        """A treenode is just a pointer back to its parent and an associated state."""
        super(TreeNode, self).__init__(state)
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
        p_sample_goal: float = 0.1,
        goal_tolerance: float = 1e-4,
    ):
        """Create RRT planner with configuration"""
        super(RRT, self).__init__(space, validate_fn)
        self.p_sample_goal = p_sample_goal
        self.reset()

    def reset(self):
        self.start_time = None
        self.goal_state = None
        self.nodes = []

    def plan(self, start, goal, num_iter: int = 1000) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py
        """
        self.start_time = time.time()
        if not self.validate(start):
            return PlanResult(False)

        # TODO: currently not supporting goal samplers
        # if callable(goal):
        #    self.sample_goal = goal
        # else:
        #    # We'll assume this is valid
        #    self.sample_goal = lambda: goal
        self.goal_state = goal
        for i in range(self.max_iterations):
            res = self.step_planner()
            if res.success:
                return res
        return PlanResult(False)

    def step_planner(self, max_time, force_sample_goal=False) -> PlanResult:
        """Continue planning for a while. In case you want to try for anytime planning."""
        assert (
            self.sample_goal is not None
        ), "no goal provided with a call to plan(start, goal)"
        assert (
            self.start_time is not None
        ), "does not look like you started planning with plan(start, goal)"

        if force_sample_goal:
            should_sample_goal = True
        else:
            should_sample_goal = random() < self.p_sample_goal
        # Get a new state
        next_state = self.sample_goal() if should_sample_goal else self.space.sample()
        closest = self.space.closest_node_to_state(next_state, self.nodes)
        for step_state in self.space.extend(closest.state, next_state):
            if not self.validate(step_state):
                # This did not work
                break
            else:
                # Create a new TreeNode poining back to closest node
                closest = TreeNode(step_state, closest)
                self.nodes.append(closest)
            # Check to see if it's the goal
            if self.space.distance(closest, self.goal_state) < self.goal_tolerance:
                # We made it! We're close enough to goal to be done
                return PlanResult(True, self.nodes[-1].backup())
        return PlanResult(False)
