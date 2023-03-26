# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu

# from home_robot.core.abstract_agent import Agent
from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.navigation_planner.rrt import RRTPlanner

from .objectnav_agent_module import ObjectNavAgentModule


class SamplingBasedObjectNavAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0):
        super(SamplingBasedObjectNavAgent, self).__init__(config, device_id)
        self.planner = RRTPlanner()
        self.voxel_map = SparseVoxelMap()

    def reset(self):
        """Clear information in the voxel map"""
        self.reset_vectorized()
        self.voxel_map.reset()
        self.planner.reset()
        self.episode_panorama_start_steps = self.panorama_start_steps

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Use this action to move around in the world"""

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            recep_goal_category,
            goal_name,
            camera_pose,
        ) = self._preprocess_obs(obs)

        raise NotImplementedError()

        return None, None
