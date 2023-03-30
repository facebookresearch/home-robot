# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from home_robot.mapping.voxel import SparseVoxelMap


class BaseState:
    def __init__(self, xy: np.ndarray, theta: float = None):
        """Create a simple state that we can interpolate between. If theta is none, we don't care about it."""
        self.xy = xy
        self.theta = theta


class TreeNode:
    """Node in an RRT sampling tree. Tracks its parent so that we can plan forwards to find a goal."""

    def __init__(self):
        raise RuntimeError("Tried to instantiate placeholder object.")


class TreeNode:
    """Node in an RRT sampling tree. Tracks its parent so that we can plan forwards to find a goal."""

    def __init__(self, state, parent: TreeNode = None):
        self.parent = parent
        self.state = state


class RRTPlanner:
    """Sampling-based planner."""

    def __init__(self):
        """Set up default params for interpolating states, sampling rates, etc."""
        raise NotImplementedError()

    def solve(self, voxel_map: SparseVoxelMap, goal: BaseState):
        """Solve the problem. Extract flattened map from sparse voxel map, then go on."""
        print(voxel_map)
        print("goal =", goal)
        raise NotImplementedError()

    def reset(self):
        """Clear out state information from the planner"""
        raise NotImplementedError()
