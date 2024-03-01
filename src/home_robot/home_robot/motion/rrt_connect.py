# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on Caelan Garrett's code from here: https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_connect.py

import time
from random import random
from typing import Callable, List

import numpy as np

from home_robot.motion.base import Planner, PlanResult
from home_robot.motion.rrt import RRT, TreeNode
from home_robot.motion.space import ConfigurationSpace, Node


class RRTConnect(RRT):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Create RRT planner with configuration"""
        super(RRTConnect, self).__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.start_time = None
        self.goal_state = None
        self.nodes_fwd = []
        self.nodes_rev = []
        self.nodes = None

    def plan(self, start, goal, verbose: bool = False) -> PlanResult:
        """Plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_connect.py
        """

        self.start_time = time.time()
        # TODO: support goal sets
        self.goal_state = goal
        # Make sure we can actually start from this position
        if not self.validate(start):
            return PlanResult(False, reason="invalid start")
        # Add start to the tree
        self.nodes_fwd.append(TreeNode(start))
        # Make sure the goal is reasonable too
        if not self.validate(goal):
            return PlanResult(False, reason="invalid goal")
        # Add start to the tree
        self.nodes_rev.append(TreeNode(goal))

        # First step - just run the RRT algo
        res, _ = self.step_planner(force_sample_goal=True, nodes=self.nodes_fwd)
        # Update the cached nodes for this planner
        self.nodes = self.nodes_fwd
        if res.success:
            return res

        for i in range(self.max_iter):
            # Loop for a certain number of iterations
            swap = i % 2 == 1
            if swap:
                nodes0, nodes1 = self.nodes_rev, self.nodes_fwd
            else:
                nodes0, nodes1 = self.nodes_fwd, self.nodes_rev
            # Sample a random point and try to connect both trees
            next_state = self.space.sample()
            # If they both connect, you won!
            res0, closest_node = self.step_planner(nodes=nodes0, next_state=next_state)
            res1, final_node = self.step_planner(
                nodes=nodes1, next_state=closest_node.state
            )
            if res1.success:
                # We found a path! Now we just need to extract it
                path1 = closest_node.backup()
                path2 = final_node.backup()
                if swap:
                    path_fwd = path2
                    path_rev = path1
                else:
                    path_fwd = path1
                    path_rev = path2
                # Update nodes cache
                self.nodes = self.nodes_fwd
                parent = path_fwd[-1]
                # Add reverse path into the tree
                for node in reversed(path_rev):
                    new_node = TreeNode(node.state, parent)
                    self.nodes.append(new_node)
                    path_fwd.append(new_node)
                    parent = new_node
                return PlanResult(True, path_fwd, planner=self)
        return PlanResult(False, planner=self)
