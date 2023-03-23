#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from home_robot.utils.config import get_control_config
from home_robot.utils.geometry.nav import xyt_base_to_global, xyt_global_to_base

from .feedback.velocity_controllers import DDVelocityControlNoplan

log = logging.getLogger(__name__)

DEFAULT_CFG_NAME = "noplan_velocity_sim"


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        cfg: Optional["DictConfig"] = None,
    ):
        if cfg is None:
            cfg = get_control_config(DEFAULT_CFG_NAME)

        # Control module
        self.control = DDVelocityControlNoplan(cfg)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True
        self._is_done = False

    def update_pose_feedback(self, xyt_current: np.ndarray):
        self.xyt_loc = xyt_current
        self._is_done = False

    def update_goal(self, xyt_goal: np.ndarray, relative: bool = False):
        self._is_done = False
        if relative:
            self.xyt_goal = xyt_base_to_global(xyt_goal, self.xyt_loc)
        else:
            self.xyt_goal = xyt_goal

    def set_yaw_tracking(self, value: bool):
        self._is_done = False
        self.track_yaw = value

    def _compute_error_pose(self):
        """
        Updates error based on robot localization
        """
        xyt_err = xyt_global_to_base(self.xyt_goal, self.xyt_loc)
        if not self.track_yaw:
            xyt_err[2] = 0.0
        else:
            xyt_err[2] = (xyt_err[2] + np.pi) % (2 * np.pi) - np.pi

        return xyt_err

    def is_done(self) -> bool:
        """Tell us if this is done and has reached its goal."""
        return self._is_done

    def compute_control(self):
        # Get state estimation
        xyt_err = self._compute_error_pose()

        # Compute control
        v_cmd, w_cmd, done = self.control(xyt_err)
        self._is_done = done

        return v_cmd, w_cmd
