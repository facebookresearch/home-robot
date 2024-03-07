# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on Caelan Garrett's code from here: https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py


import time
from random import random
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
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
        max_iter: int = 100,
    ):
        """Create RRT planner with configuration"""
        super(RRT, self).__init__(space, validate_fn)
        self.p_sample_goal = p_sample_goal
        self.goal_tolerance = goal_tolerance
        self.max_iter = max_iter
        self.reset()

    def reset(self):
        self.start_time = None
        self.goal_state = None
        self.nodes = []

    def plan(self, start, goal, verbose: bool = True) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py
        """
        assert len(start) == self.space.dof, "invalid start dimensions"
        assert len(goal) == self.space.dof, "invalid goal dimensions"
        self.reset()
        self.start_time = time.time()
        if not self.validate(start):
            if verbose:
                print("[Planner] invalid start")
            return PlanResult(False)
        if not self.validate(goal):
            if verbose:
                print("[Planner] invalid goal")
            return PlanResult(False)
        # Add start to the tree
        self.nodes.append(TreeNode(start))

        # TODO: currently not supporting goal samplers
        # if callable(goal):
        #    self.sample_goal = goal
        # else:
        #    # We'll assume this is valid
        #    self.sample_goal = lambda: goal
        self.goal_state = goal
        # Always try goal first
        res, _ = self.step_planner(force_sample_goal=True)
        if res.success:
            return res
        # Iterate a bunch of times
        for i in range(self.max_iter - 1):
            res, _ = self.step_planner(nodes=self.nodes)
            if res.success:
                return res
        return PlanResult(False)

    def step_planner(
        self,
        force_sample_goal=False,
        nodes: Optional[TreeNode] = None,
        next_state: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Continue planning for a while. In case you want to try for anytime planning."""
        assert (
            self.goal_state is not None
        ), "no goal provided with a call to plan(start, goal)"
        assert (
            self.start_time is not None
        ), "does not look like you started planning with plan(start, goal)"

        if force_sample_goal or next_state is not None:
            should_sample_goal = True
        else:
            should_sample_goal = random() < self.p_sample_goal

        # Get a new state
        if next_state is not None:
            goal_state = next_state
        else:
            goal_state = self.goal_state
        # Set the state we will try to move to
        if next_state is None:
            next_state = goal_state if should_sample_goal else self.space.sample()
        closest = self.space.closest_node_to_state(next_state, nodes)
        for step_state in self.space.extend(closest.state, next_state):
            if not self.validate(step_state):
                # This did not work
                break
            else:
                # Create a new TreeNode poining back to closest node
                closest = TreeNode(step_state, parent=closest)
                nodes.append(closest)
            # Check to see if it's the goal
            if self.space.distance(nodes[-1].state, goal_state) < self.goal_tolerance:
                # We made it! We're close enough to goal to be done
                return PlanResult(True, nodes[-1].backup()), nodes[-1]
        return PlanResult(False), closest
