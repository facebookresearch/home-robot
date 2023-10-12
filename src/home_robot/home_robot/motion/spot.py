# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from home_robot.motion.robot import Footprint, RobotModel


class SimpleSpotKinematics(RobotModel):
    """Placeholder kinematics model for the Boston Dynamics spot robot."""

    def __init__(self, *args, **kwargs):
        """Create a Spot kinematic model, which only models the base."""
        super(SimpleSpotKinematics, self).__init__(*args, **kwargs)
        self.dof = 3

    def set_config(self, q):
        """put the robot in the right position for bullet planning"""
        raise NotImplementedError("Bullet planning not yet supported for this robot")

    def get_config(self):
        """turn current state into a vector"""
        raise NotImplementedError("Bullet planning not yet supported for this robot")

    def set_head_config(self, q):
        """just for the head"""
        raise NotImplementedError("Bullet planning not yet supported for this robot")

    def set_camera_to_head(self, camera, q=None):
        """take a bullet camera and put it on the robot's head"""
        if q is not None:
            self.set_head_config(q)
        raise NotImplementedError("Bullet planning not yet supported for this robot")

    def get_footprint(self) -> np.ndarray:
        """return a footprint mask that we can check 2d collisions against"""
        # The dimensions of spot are 0.5m x 1.1m (https://dev.bostondynamics.com/docs/concepts/about_spot)
        # Currently taking half because the planner is to conservative
        return Footprint(width=0.5, length=1.1, width_offset=0.0, length_offset=0.0)
