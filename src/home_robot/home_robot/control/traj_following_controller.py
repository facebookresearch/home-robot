# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading

import numpy as np

from home_robot.utils.geometry.nav import xyt_global_to_base

V_MAX = 0.2  # base.params["motion"]["default"]["vel_m"]
W_MAX = 0.45  # (vel_m_max - vel_m_default) / wheel_separation_m
K_P = 0.5
DAMP_RATIO = 1.0
DECAY = 0.5


class TrajFollower:
    def __init__(self, is_open_loop=False):
        self._is_open_loop = is_open_loop

        # Compute gain
        self.kp = K_P
        self.ki = (DAMP_RATIO * self.kp) ** 2 / 4.0

        # Init
        self._traj_update_lock = threading.Lock()

        self._is_done = True
        self.traj = None
        self.traj_buffer = None

        self.e_int = np.zeros(3)
        self._t_prev = 0

    def update_trajectory(self, traj):
        with self._traj_update_lock:
            self.traj_buffer = traj

    def is_done(self):
        return self._is_done

    def forward(self, xyt, t):
        """Returns velocity control command (v, w)"""
        # Check for trajectory updates
        if self.traj_buffer is not None:
            with self._traj_update_lock:
                self.traj = self.traj_buffer
                self.traj_buffer = None
            self._is_done = False

        # Return zero velocites if no trajectory is active
        if self._is_done:
            return 0.0, 0.0

        # Query trajectory for desired states
        xyt_traj, dxyt_traj, done = self.traj(t)
        if done:
            self._is_done = True

        # Feedback control
        dt = t - self._t_prev
        self._t_prev = t
        v, w = self._feedback_controller(xyt_traj, dxyt_traj, xyt, dt)

        return v, w

    def _feedback_controller(self, xyt_des, dxyt_des, xyt_curr, dt):
        # Compute reference input
        u_ref = np.array([np.linalg.norm(dxyt_des[:2]), dxyt_des[2]])

        if self._is_open_loop:
            u_output = u_ref

        else:
            # Compute error in local frame
            e = xyt_global_to_base(xyt_des, xyt_curr)

            # Compute desired error derivative via PI control
            self.e_int = DECAY * self.e_int + e * dt
            de_des = -self.kp * e - self.ki * self.e_int

            # Compute velocity feedback commands to achieve desired error derivative
            M_u2e = np.array([[-1, e[1]], [0, -e[0]], [0, -1]])
            M_ur2e = np.array([[np.cos(e[2]), 0], [np.sin(e[2]), 0], [0, 1]])
            u_output = np.linalg.pinv(M_u2e) @ (de_des - M_ur2e @ u_ref)

        return u_output[0], u_output[1]
