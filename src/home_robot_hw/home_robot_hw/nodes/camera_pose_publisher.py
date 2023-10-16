#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import rospy
import tf
import trimesh.transformations as tra
from geometry_msgs.msg import PoseStamped

STRETCH_BASE_FRAME = "base_link"
STRETCH_CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_HEAD_CAMERA_ROTATIONS = 3

#from home_robot.motion.stretch import (
#    STRETCH_BASE_FRAME,
#    STRETCH_CAMERA_FRAME,
#    STRETCH_HEAD_CAMERA_ROTATIONS,
#)

from home_robot.utils.pose import to_matrix
from home_robot_hw.ros.utils import matrix_to_pose_msg


class CameraPosePublisher(object):
    """publishes the camera pose constantly so that we do not have a dependency on /tf"""

    def __init__(self, topic_name: str = "camera_pose"):
        self._pub = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
        self._listener = tf.TransformListener()
        self._seq = 0

    def spin(self, rate=10):
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self._listener.lookupTransform(
                    "map", STRETCH_CAMERA_FRAME, rospy.Time(0)
                )
                matrix = to_matrix(trans, rot)

                # We rotate by 90 degrees from the frame of realsense hardware since we are also rotating images to be upright
                matrix_rotated = matrix @ tra.euler_matrix(0, 0, -np.pi / 2)

                msg = PoseStamped(pose=matrix_to_pose_msg(matrix_rotated))
                msg.header.stamp = rospy.Time.now()
                msg.header.seq = self._seq
                self._pub.publish(msg)
                self._seq += 1
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                continue
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("camera_pose_publisher")
    publisher = CameraPosePublisher()
    publisher.spin()
