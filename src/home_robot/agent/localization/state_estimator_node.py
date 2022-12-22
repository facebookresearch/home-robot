import logging
import time
import threading
from typing import Optional

import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R
import rospy
import tf2_ros
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry

from home_robot.hw.ros.utils import matrix_to_pose_msg, matrix_from_pose_msg

log = logging.getLogger(__name__)

SLAM_CUTOFF_HZ = 0.2


def cutoff_angle(duration, cutoff_freq):
    return 2 * np.pi * duration * cutoff_freq


class NavStateEstimator:
    def __init__(self):
        # Initialize
        self._slam_inject_lock = threading.Lock()

        self._filtered_pose = sp.SE3()
        self._slam_pose_sp = sp.SE3()
        self._t_odom_prev: Optional[rospy.Time] = None
        self._pose_odom_prev = sp.SE3()

    def _publish_filtered_state(self, timestamp):
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
        t.transform.translation.x = pose_msg.translation.x
        t.transform.translation.y = pose_msg.translation.y
        t.transform.translation.z = 0
        t.transform.rotation.x = pose_msg.rotation.x
        t.transform.rotation.y = pose_msg.rotation.y
        t.transform.rotation.z = pose_msg.rotation.z
        t.transform.rotation.w = pose_msg.rotation.w

        self._tf_broadcaster.sendTransform(t)

    def _odom_callback(self, pose: Odometry):
        t_curr = rospy.Time.now()

        # Compute injected signals into filtered pose
        pose_odom = sp.SE3(matrix_from_pose_msg(pose.pose.pose))
        pose_diff_odom = self._pose_odom_prev.inverse() * pose_odom
        with self._slam_inject_lock:
            pose_diff_slam = self._filtered_pose.inverse() * self._slam_pose_sp

        # Update filtered pose
        if self._t_odom_prev is None:
            self._t_odom_prev = t_curr
        t_interval_secs = (t_curr - self._t_odom_prev).to_sec()
        w = cutoff_angle(t_interval_secs, SLAM_CUTOFF_HZ)
        coeff = 1 / (w + 1)

        pose_diff_log = (
            coeff * pose_diff_odom.log() + (1 - coeff) * pose_diff_slam.log()
        )
        self._filtered_pose = self._filtered_pose * sp.SE3.exp(pose_diff_log)
        self._publish_filtered_state(pose.header.stamp)

        # Update variables
        self._pose_odom_prev = pose_odom
        self._t_odom_prev = t_curr

    def _slam_pose_callback(self, pose: PoseWithCovarianceStamped):
        # Update slam pose for filtering
        with self._slam_inject_lock:
            self._slam_pose_sp = sp.SE3(matrix_from_pose_msg(pose.pose.pose))

    def run(self):
        # ROS comms
        rospy.init_node("state_estimator")

        self._estimator_pub = rospy.Publisher(
            "state_estimator/pose_filtered", PoseStamped, queue_size=1
        )
        self._world_frame_id = "odom"
        self._base_frame_id = "base_link_estimator"
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.Subscriber(
            "poseupdate",
            PoseWithCovarianceStamped,
            self._slam_pose_callback,
            queue_size=1,
        )  # This comes from hector_slam. It's a transform from src_frame = 'base_link', target_frame = 'map'
        rospy.Subscriber(
            "odom", Odometry, self._odom_callback, queue_size=1
        )  # This comes from wheel odometry.

        # Run
        log.info("State Estimator launched.")
        rospy.spin()


if __name__ == "__main__":
    node = NavStateEstimator()
    node.run()
