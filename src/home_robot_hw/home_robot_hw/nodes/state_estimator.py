#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import threading
import time
from typing import Optional

import numpy as np
import rospy
import sophus as sp
import tf2_ros
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

from home_robot_hw.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg

log = logging.getLogger(__name__)

SLAM_CUTOFF_HZ = 0.2


def cutoff_angle(duration, cutoff_freq):
    return 2 * np.pi * duration * cutoff_freq


class NavStateEstimator:
    def __init__(self, trust_slam=False):
        """Create nav state estimator.

        trust_slam: just use the slam pose instead of odometry.
        """
        self._trust_slam = trust_slam
        self._use_history = True

        # Create a lock to handle thread safety for pose updates
        self._slam_inject_lock = threading.Lock()

        self._filtered_pose = sp.SE3()
        self._slam_pose_sp = sp.SE3()
        self._slam_pose_prev = sp.SE3()
        self._t_odom_prev: Optional[rospy.Time] = None
        self._pose_odom_prev = sp.SE3()

    def _publish_filtered_state(self, timestamp):
        if self._trust_slam:
            pose_msg = matrix_to_pose_msg(self._slam_pose_sp.matrix())
        else:
            pose_msg = matrix_to_pose_msg(self._filtered_pose.matrix())

        # Publish pose msg
        pose_out = PoseStamped()
        pose_out.header.stamp = timestamp
        pose_out.pose = pose_msg

        self._estimator_pub.publish(pose_out)

        # Publish to tf2
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self._world_frame_id
        t.child_frame_id = self._base_frame_id
        t.transform.translation.x = pose_msg.position.x
        t.transform.translation.y = pose_msg.position.y
        t.transform.translation.z = 0
        t.transform.rotation.x = pose_msg.orientation.x
        t.transform.rotation.y = pose_msg.orientation.y
        t.transform.rotation.z = pose_msg.orientation.z
        t.transform.rotation.w = pose_msg.orientation.w

        self._tf_broadcaster.sendTransform(t)

    def _odom_callback(self, pose: Odometry):
        t_curr = rospy.Time.now()

        # Compute injected signals into filtered pose
        pose_odom = sp.SE3(matrix_from_pose_msg(pose.pose.pose))
        pose_diff_odom = self._pose_odom_prev.inverse() * pose_odom
        with self._slam_inject_lock:
            if not self._use_history:
                pose_diff_slam = self._slam_pose_prev.inverse() * self._slam_pose_sp
                slam_pose = self._slam_pose_prev.copy()
            else:
                pose_diff_slam = self._filtered_pose.inverse() * self._slam_pose_sp
                slam_pose = self._filtered_pose

        # Update filtered pose
        if self._t_odom_prev is None:
            self._t_odom_prev = t_curr
        t_interval_secs = (t_curr - self._t_odom_prev).to_sec()
        w = cutoff_angle(t_interval_secs, SLAM_CUTOFF_HZ)
        coeff = 1 / (w + 1)

        pose_diff_log = (
            coeff * pose_diff_odom.log() + (1 - coeff) * pose_diff_slam.log()
        )
        self._filtered_pose = slam_pose * sp.SE3.exp(pose_diff_log)
        self._publish_filtered_state(pose.header.stamp)

        # Update variables
        self._pose_odom_prev = pose_odom
        self._t_odom_prev = t_curr

    def _slam_pose_callback(self, pose: PoseWithCovarianceStamped) -> None:
        """Update slam pose for filtering"""
        with self._slam_inject_lock:
            self._slam_pose_prev = self._slam_pose_sp
            self._slam_pose_sp = sp.SE3(matrix_from_pose_msg(pose.pose.pose))

    def run(self):
        # Create publishers and subscribers
        self._estimator_pub = rospy.Publisher(
            "state_estimator/pose_filtered", PoseStamped, queue_size=1
        )
        self._world_frame_id = "map"
        # TODO: if we need to debug this vs. the scan matcher
        # self._base_frame_id = "base_link_estimator"
        self._base_frame_id = "base_link"
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()
    
        # This comes from hector_slam.
        # It's a transform from src_frame = 'base_link', target_frame = 'map'
        # The *inverse* is published by default from hector as the transform from map to base -
        # you can verify this with:
        #   rosrun tf tf_echo map base_link
        # Which will show the same output as this topic.
        rospy.Subscriber(
            "poseupdate",
            PoseWithCovarianceStamped,
            self._slam_pose_callback,
            queue_size=1,
        )
        # This pose update comes from wheel odometry
        rospy.Subscriber(
            "odom", Odometry, self._odom_callback, queue_size=1
        )

        # Run
        log.info("State Estimator launched.")
        rospy.spin()


if __name__ == "__main__":
    # Init ros node outside of run function
    rospy.init_node("state_estimator")
    # Create the state estimator
    node = NavStateEstimator()
    node.run()
