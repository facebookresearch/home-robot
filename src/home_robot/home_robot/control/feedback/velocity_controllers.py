# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from home_robot.utils.geometry import normalize_ang_error


class DiffDriveVelocityController(abc.ABC):
    """
    Abstract class for differential drive robot velocity controllers.
    """

    def set_linear_error_tolerance(self, error_tol: float):
        self.lin_error_tol = error_tol

    def set_angular_error_tolerance(self, error_tol: float):
        self.ang_error_tol = error_tol

    @abc.abstractmethod
    def __call__(self, xyt_err: np.ndarray) -> Tuple[float, float, bool]:
        """Contain execution logic, predict velocities for the left and right wheels. Expected to
        return true/false if we have reached this goal and the controller will be moving no
        farther."""
        pass


class DDVelocityControlNoplan(DiffDriveVelocityController):
    """
    Control logic for differential drive robot velocity control.
    Does not plan at all, instead uses heuristics to gravitate towards the goal.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.reset_error_tolerances()
        self.reset_velocity_profile()

    def reset_velocity_profile(self):
        """Read velocity configuration info from the config"""
        self.update_velocity_profile(
            self.cfg.v_max, self.cfg.w_max, self.cfg.acc_lin, self.cfg.acc_ang
        )

    def update_velocity_profile(
        self,
        v_max: Optional[float] = None,
        w_max: Optional[float] = None,
        acc_lin: Optional[float] = None,
        acc_ang: Optional[float] = None,
    ):
        """Call controller and update velocity profile.

        Parameters:
            v_max: max linear velocity
            w_max: max rotational velocity
            acc_lin: forward acceleration
            acc_ang: rotational acceleration"""
        if v_max is not None:
            self.v_max = v_max
        if w_max is not None:
            self.w_max = w_max
        if acc_lin is not None:
            self.acc_lin = acc_lin
        if acc_ang is not None:
            self.acc_ang = acc_ang

    def reset_error_tolerances(self):
        """Reset error tolerances to default values"""
        self.lin_error_tol = self.cfg.lin_error_tol
        self.ang_error_tol = self.cfg.ang_error_tol

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max):
        """
        Computes velocity based on distance from target (trapezoidal velocity profile).
        Used for both linear and angular motion.
        """
        t = np.sqrt(2.0 * abs(x_err) / a)  # x_err = (1/2) * a * t^2
        v = min(a * t, v_max)
        return v * np.sign(x_err)

    def _turn_rate_limit(self, lin_err, heading_diff, w_max):
        """
        Compute velocity limit that prevents path from overshooting goal

        heading error decrease rate > linear error decrease rate
        (w - v * np.sin(phi) / D) / phi > v * np.cos(phi) / D
        v < (w / phi) / (np.sin(phi) / D / phi + np.cos(phi) / D)
        v < w * D / (np.sin(phi) + phi * np.cos(phi))

        (D = linear error, phi = angular error)
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0

        if heading_diff > self.cfg.max_heading_ang:
            return 0.0
        else:
            return (
                w_max
                * lin_err
                / (np.sin(heading_diff) + heading_diff * np.cos(heading_diff) + 1e-5)
            )

    def __call__(
        self, xyt_err: np.ndarray, allow_reverse: bool = False
    ) -> Tuple[float, float, bool]:
        v_cmd = w_cmd = 0
        in_reverse = False
        done = True

        # Compute errors
        lin_err_abs = np.linalg.norm(xyt_err[0:2])
        ang_err = xyt_err[2]

        heading_err = np.arctan2(xyt_err[1], xyt_err[0])

        # Check if reverse is required
        if allow_reverse and abs(heading_err) > np.pi / 2.0:
            in_reverse = True
            heading_err = normalize_ang_error(heading_err + np.pi)

        # Go to goal XY position if not there yet
        if lin_err_abs > self.lin_error_tol:
            # Compute linear velocity -- move towards goal XY
            v_raw = self._velocity_feedback_control(
                lin_err_abs, self.acc_lin, self.v_max
            )
            v_limit = self._turn_rate_limit(
                lin_err_abs,
                abs(heading_err),
                self.w_max / 2.0,
            )
            v_cmd = np.clip(v_raw, 0.0, v_limit)

            # Compute angular velocity -- turn towards goal XY
            w_cmd = self._velocity_feedback_control(
                heading_err, self.acc_ang, self.w_max
            )
            done = False

        # Rotate to correct yaw if XY position is at goal
        elif abs(ang_err) > self.ang_error_tol:
            # Compute angular velocity -- turn to goal orientation
            w_cmd = self._velocity_feedback_control(ang_err, self.acc_ang, self.w_max)
            done = False

        if in_reverse:
            v_cmd = -v_cmd

        return v_cmd, w_cmd, done
