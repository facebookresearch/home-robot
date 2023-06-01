#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from home_robot.utils.config import get_control_config
from home_robot.utils.geometry import normalize_ang_error

from .feedback.velocity_controllers import DDVelocityControlNoplan

log = logging.getLogger(__name__)

DEFAULT_CFG_NAME = "noplan_velocity_sim"


def xyt_global_to_base(xyt_world2target, xyt_world2base):
    """Transforms SE2 coordinates from global frame to local frame

    This function was created to temporarily remove dependency on sophuspy from the controller.
    TODO: Unify geometry utils across repository

    Args:
        xyt_world2target: SE2 transformation from world to target
        xyt_world2base: SE2 transformation from world to base

    Returns:
        SE2 transformation from base to target
    """
    x_diff = xyt_world2target[0] - xyt_world2base[0]
    y_diff = xyt_world2target[1] - xyt_world2base[1]
    theta_diff = xyt_world2target[2] - xyt_world2base[2]
    base_cos = np.cos(xyt_world2base[2])
    base_sin = np.sin(xyt_world2base[2])

    xyt_base2target = np.zeros(3)
    xyt_base2target[0] = x_diff * base_cos + y_diff * base_sin
    xyt_base2target[1] = x_diff * -base_sin + y_diff * base_cos
    xyt_base2target[2] = theta_diff

    return xyt_base2target


def xyt_base_to_global(xyt_base2target, xyt_world2base):
    """Transforms SE2 coordinates from local frame to global frame

    This function was created to temporarily remove dependency on sophuspy from the controller.
    TODO: Unify geometry utils across repository

    Args:
        xyt_base2target: SE2 transformation from base to target
        xyt_world2base: SE2 transformation from world to base

    Returns:
        SE2 transformation from world to target
    """
    base_cos = np.cos(xyt_world2base[2])
    base_sin = np.sin(xyt_world2base[2])
    x_base2target_global = xyt_base2target[0] * base_cos - xyt_base2target[1] * base_sin
    y_base2target_global = xyt_base2target[0] * base_sin + xyt_base2target[1] * base_cos

    xyt_world2target = np.zeros(3)
    xyt_world2target[0] = xyt_world2base[0] + x_base2target_global
    xyt_world2target[1] = xyt_world2base[1] + y_base2target_global
    xyt_world2target[2] = xyt_world2base[2] + xyt_base2target[2]

    return xyt_world2target


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        cfg: Optional["DictConfig"] = None,
        verbose=False,
    ):
        if cfg is None:
            cfg = get_control_config(DEFAULT_CFG_NAME)
        self.cfg = cfg
        self._timeout = self.cfg.timeout

        # Control module
        self.control = DDVelocityControlNoplan(cfg)
        self.update_velocity_profile(
            self.cfg.v_max, self.cfg.w_max, self.cfg.acc_lin, self.cfg.acc_ang
        )

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True
        self._is_done = False

        self.verbose = verbose

    def update_velocity_profile(
        self,
        v_max: Optional[float] = None,
        w_max: Optional[float] = None,
        acc_lin: Optional[float] = None,
        acc_ang: Optional[float] = None,
    ):
        """Call controller and update velocity profile"""
        self.control.update_velocity_profile(v_max, w_max, acc_lin, acc_ang)

    def update_pose_feedback(self, xyt_current: np.ndarray):
        self.xyt_loc = xyt_current
        self._is_done = False

    def compute_current_error(self) -> np.ndarray:
        """Compute xyt error from location to goal"""
        xyt_err = xyt_global_to_base(self.xyt_goal, self.xyt_loc)

        # Normalize angular error to between -pi and pi
        xyt_err[2] = normalize_ang_error(xyt_err[2])
        return xyt_err

    def update_goal(self, xyt_goal: np.ndarray, relative: bool = False):
        self._is_done = False
        if relative:
            self.xyt_goal = xyt_base_to_global(xyt_goal, self.xyt_loc)
        else:
            self.xyt_goal = xyt_goal

        # Compute error in order to get dynamic target thresholds for low-level controller
        print("...... updated goal")
        xyt_err = self.compute_current_error()
        lin_err = np.linalg.norm(xyt_err[:2])
        if lin_err > self.cfg.lin_error_tol or abs(xyt_err[2]) > self.cfg.ang_error_tol:
            self.control.set_linear_error_tolerance(self.cfg.lin_error_tol)
            self.control.set_angular_error_tolerance(self.cfg.ang_error_tol)
        else:
            print(
                f"WARNING: sent a goal with lower distance than target error tolerance! Linear err = {lin_err}, Angular error = {xyt_err[2]}"
            )
            new_lin_tol = max(
                self.cfg.min_lin_error_tol, self.cfg.lin_error_ratio * lin_err
            )
            print(f" -> setting linear tolerance to {new_lin_tol}")
            self.control.set_linear_error_tolerance(new_lin_tol)
            new_ang_tol = max(
                self.cfg.min_ang_error_tol, self.cfg.ang_error_ratio * xyt_err[2]
            )
            print(f" -> setting angular tolerance to {new_ang_tol}")
            self.control.set_angular_error_tolerance(new_ang_tol)

    def set_yaw_tracking(self, value: bool):
        self._is_done = False
        self.track_yaw = value

    def _compute_error_pose(self) -> np.ndarray:
        """
        Updates error based on robot localization
        """
        xyt_err = self.compute_current_error()

        # Set angular error to 0 if not tracking target yaw
        if not self.track_yaw:
            xyt_err[2] = 0.0

        return xyt_err

    def is_done(self) -> bool:
        """Tell us if this is done and has reached its goal."""
        return self._is_done

    def timeout(self, time_taken: float) -> bool:
        """Returns true if it's taken too long."""
        return time_taken > self._timeout

    def compute_control(self) -> Tuple[float, float]:
        # Get state estimation
        xyt_err = self._compute_error_pose()
        lin_err = np.linalg.norm(xyt_err[:2])

        # Move backwards if conditions are met
        allow_reverse = False
        if np.linalg.norm(xyt_err[:2]) < self.cfg.max_rev_dist:
            allow_reverse = True

        # Compute control
        v_cmd, w_cmd, done = self.control(xyt_err, allow_reverse=allow_reverse)
        self._is_done = done

        if self.verbose:
            print(
                " - err =", lin_err, xyt_err[2], "done =", done, "cmd =", v_cmd, w_cmd
            )

        return v_cmd, w_cmd
