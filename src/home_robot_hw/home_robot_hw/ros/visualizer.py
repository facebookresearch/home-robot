# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from home_robot_hw.ros.utils import matrix_to_pose_msg


class Visualizer(object):
    """Simple visualizer to send a single marker message"""

    def __init__(self, topic_name: str, rgba: Optional[List] = None):
        self.pub = rospy.Publisher(topic_name, Marker, queue_size=1)
        if rgba is None:
            rgba = [1, 0, 0, 0.75]
        self.rgba = rgba

    def __call__(self, pose_matrix: np.ndarray, frame_id: str = "map"):
        """publish 3D pose as a marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.ARROW
        marker.pose = matrix_to_pose_msg(pose_matrix)
        marker.color.r = self.rgba[0]
        marker.color.g = self.rgba[1]
        marker.color.b = self.rgba[2]
        marker.color.a = self.rgba[3]
        marker.scale.x = 0.2
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        self.pub.publish(marker)

    def publish_2d(self, pose_matrix: np.ndarray, frame_id: str = "map"):
        """Publish a 2D pose as a marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.SPHERE
        marker.pose = matrix_to_pose_msg(pose_matrix)
        marker.color.r = self.rgba[0]
        marker.color.g = self.rgba[1]
        marker.color.b = self.rgba[2]
        marker.color.a = self.rgba[3]
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        self.pub.publish(marker)


class ArrayVisualizer(object):
    """Simple visualizer to send an array of marker message"""

    def __init__(self, topic_name: str, rgba: Optional[List] = None):
        self.array_pub = rospy.Publisher(topic_name, MarkerArray, queue_size=1)
        if rgba is None:
            rgba = [1, 0, 0, 0.75]
        self.rgba = rgba

    def __call__(self, pose_matrix_array: np.ndarray, frame_id: str = "map"):
        markers = MarkerArray()
        i = 0
        for pose_matrix in pose_matrix_array:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.id = i
            i += 1
            marker.type = Marker.ARROW
            marker.pose = matrix_to_pose_msg(pose_matrix)
            marker.color.r = self.rgba[0]
            marker.color.g = self.rgba[1]
            marker.color.b = self.rgba[2]
            marker.color.a = self.rgba[3]
            marker.scale.x = 0.2
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            markers.markers.append(marker)
        self.array_pub.publish(markers)
