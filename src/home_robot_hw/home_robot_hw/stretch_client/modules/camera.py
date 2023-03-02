# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import numpy as np
import rospy
import trimesh.transformations as tra

from home_robot_hw.ros.camera import RosCamera

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


class StretchCameraInterface:
    def __init__(
        self,
        ros_client,
        init_cameras: bool = True,
        color_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        depth_buffer_size: Optional[int] = None,
    ):
        self._ros_client = ros_client
        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        self._depth_buffer_size = depth_buffer_size

        # Init cameras
        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras(color_topic, depth_topic)
            self._wait_for_cameras()

    # Interface methods

    def get_pose(self, rotated=False):
        """get matrix version of the camera pose"""
        mat = self._t_camera_pose.matrix()
        if rotated:
            # If we are using the rotated versions of the images
            return mat @ tra.euler_matrix(0, 0, -np.pi / 2)
        else:
            return mat

    def set_pan_tilt(self, pan: Optional[float] = None, tilt: Optional[float] = None):
        joint_goals = {}
        if pan is not None:
            joint_goals[ROS_HEAD_PAN] = pan
        if tilt is not None:
            joint_goals[ROS_HEAD_TILT] = tilt

        self._ros_client.send_ros_trajectory_goals(joint_goals)

    def process_depth(self, depth):
        depth[depth < self.min_depth_val] = MIN_DEPTH_REPLACEMENT_VALUE
        depth[depth > self.max_depth_val] = MAX_DEPTH_REPLACEMENT_VALUE
        return depth

    def get_images(self, compute_xyz=False, rotate_images=True):
        """helper logic to get images from the robot's camera feed"""
        rgb = self.rgb_cam.get()
        if self.filter_depth:
            dpt = self.dpt_cam.get_filtered()
        else:
            dpt = self.process_depth(self.dpt_cam.get())
        if compute_xyz:
            xyz = self.dpt_cam.depth_to_xyz(self.dpt_cam.fix_depth(dpt))
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

    def _create_cameras(self, color_topic=None, depth_topic=None):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(self._color_topic)
        self.dpt_cam = RosCamera(self._depth_topic, buffer_size=self._depth_buffer_size)
        self.filter_depth = self._depth_buffer_size is not None

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        print("rgb frame =", self.rgb_cam.get_frame())
        print("dpt frame =", self.dpt_cam.get_frame())
        if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
            raise RuntimeError("issue with camera setup; depth and rgb not aligned")
