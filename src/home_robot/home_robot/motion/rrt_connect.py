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

    def plan(self, start, goal) -> PlanResult:
        """Plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_connect.py
        """

        self.start_time = time.time()
        # Make sure we can actually start from this position
        if not self.validate(start):
            return PlanResult(False)
        # Add start to the tree
        self.nodes_fwd.append(TreeNode(start))
        # Make sure the goal is reasonable too
        if not self.validate(goal):
            return PlanResult(False)
        # Add start to the tree
        self.nodes_rev.append(TreeNode(goal))

        force_sample_goal = True
        for i in range(self.max_iter):
            # Loop for a certain number of iterations
            if force_sample_goal:
                # Always sample in the first timestep
                should_sample_goal = True
                force_sample_goal = False
            else:
                should_sample_goal = random() < self.p_sample_goal

            print(should_sample_goal)

        return PlanResult(False)
