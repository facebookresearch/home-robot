# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import numpy as np
import rospy
import trimesh.transformations as tra

from home_robot.motion.robot import Robot
from home_robot.motion.stretch import HelloStretchIdx

from .abstract import AbstractControlModule

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


class StretchHeadClient(AbstractControlModule):
    min_depth_val = 0.1
    max_depth_val = 4.0

    def __init__(
        self,
        ros_client,
        robot_model: Robot,
    ):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model

    # Interface methods

    def get_pose(self, rotated=False):
        """get matrix version of the camera pose"""
        mat = self._ros_client.se3_camera_pose.matrix()
        if rotated:
            # If we are using the rotated versions of the images
            return mat @ tra.euler_matrix(0, 0, -np.pi / 2)
        else:
            return mat

    def get_pan_tilt(self) -> Tuple[float, float]:
        q, _, _ = self._ros_client.get_joint_state()
        return q[HelloStretchIdx.HEAD_PAN], q[HelloStretchIdx.HEAD_TILT]

    def set_pan_tilt(
        self,
        pan: Optional[float] = None,
        tilt: Optional[float] = None,
        blocking: bool = True,
    ):
        joint_goals = {}
        if pan is not None:
            joint_goals[self._ros_client.HEAD_PAN] = pan
        if tilt is not None:
            joint_goals[self._ros_client.HEAD_TILT] = tilt

        self._ros_client.send_trajectory_goals(joint_goals)

        self._register_wait(self._ros_client.wait_for_trajectory_action)
        if blocking:
            self.wait()

    def look_at_ee(self, blocking: bool = True):
        """Point camera sideways towards the gripper"""
        pan, tilt = self._robot_model.look_at_ee
        self.set_pan_tilt(pan, tilt, blocking=blocking)

    def look_front(self, blocking: bool = True):
        """Point camera forwards at a 45-degree downwards angle"""
        pan, tilt = self._robot_model.look_front
        self.set_pan_tilt(pan, tilt, blocking=blocking)

    def look_ahead(self, blocking: bool = True):
        """Point camera forwards horizontally"""
        pan, tilt = self._robot_model.look_ahead
        self.set_pan_tilt(pan, tilt, blocking=blocking)

    def get_images(self, compute_xyz=False, rotate_images=True):
        """helper logic to get images from the robot's camera feed"""
        rgb = self._ros_client.rgb_cam.get()
        if self._ros_client.filter_depth:
            dpt = self._ros_client.dpt_cam.get_filtered()
        else:
            dpt = self._process_depth(self._ros_client.dpt_cam.get())
        if compute_xyz:
            xyz = self._ros_client.dpt_cam.depth_to_xyz(
                self._ros_client.dpt_cam.fix_depth(dpt)
            )
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt]
            xyz = None

        if rotate_images:
            # Get xyz in base coords for later
            imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]

        if xyz is not None:
            xyz = imgs[-1]
            H, W = rgb.shape[:2]
            xyz = xyz.reshape(-1, 3)

            if rotate_images:
                # Rotate the stretch camera so that top of image is "up"
                R_stretch_camera = tra.euler_matrix(0, 0, -np.pi / 2)[:3, :3]
                xyz = xyz @ R_stretch_camera
                xyz = xyz.reshape(H, W, 3)
                imgs[-1] = xyz

        return imgs

    # Helper methods

    def _process_depth(self, depth):
        depth[depth < self.min_depth_val] = MIN_DEPTH_REPLACEMENT_VALUE
        depth[depth > self.max_depth_val] = MAX_DEPTH_REPLACEMENT_VALUE
        return depth

    def _enable_hook(self) -> bool:
        """Dummy override for abstract method"""
        pass

    def _disable_hook(self) -> bool:
        """Dummy override for abstract method"""
        pass
