# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading
from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from home_robot.utils.image import Camera
from home_robot_hw.ros.msg_numpy import image_to_numpy

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"
DEFAULT_LIDAR_TOPIC = "/scan"
DEFAULT_POSE_TOPIC = "/state_estimator/pose_filtered"


class SynchronizedSensors(object):
    """Quick class to use a time synchronizer to collect sensor data to speed up the robot execution."""

    def _process_laser(self, scan_msg):

        # Get the range and angle data from the scan message
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        # convert polar coordinates (ranges, angles) to Cartesian coordinates (x, y)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # Stack x and y coordinates to create a 2D NumPy array of points
        lidar_points = np.column_stack((xs, ys))
        return lidar_points

    def _start_camera(self, name):
        camera_info_topic = name + "/camera_info"
        if self.verbose:
            print("Waiting for camera info on", self._camera_info_topic + "...")
        cam_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        topic = name + "/image_raw"
        return Subscriber(topic, Image), cam_info

    def _callback(self, color, depth, lidar, pose):
        """Process the data and expose it"""
        self._lidar_points = self._process_laser(lidar)
        self._times = {
            "rgb": color.header.stamp.to_sec(),
            "depth": depth.header.stamp.to_sec(),
            "lidar": lidar.header.stamp.to_sec(),
            "pose": pose.header.stamp.to_sec(),
        }

    def get_times(self) -> Dict[str, float]:
        """Get the times for all measurements"""
        return self._times

    def __init__(
        self,
        color_name,
        depth_name,
        scan_topic,
        pose_topic,
        verbose=False,
        slop_time_seconds=0.05,
    ):
        self.verbose = verbose
        self._t = rospy.Time(0)
        self._lock = threading.Lock()

        if verbose:
            print("Creating subs...")
        self._color_sub, self._color_camera_info = self._start_camera(color_name)
        self._depth_sub, self._depth_camera_info = self._start_camera(depth_name)
        self._lidar_sub = Subscriber(scan_topic, LaserScan)
        self._pose_sub = Subscriber(pose_topic, PoseStamped)

        # Store time information
        self._times = {}

        if verbose:
            print("Time synchronizer...")
        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._lidar_sub, self._pose_sub],
            queue_size=10,
            slop=slop_time_seconds,
        )
        self._sync.registerCallback(self._callback)


if __name__ == "__main__":
    rospy.init_node("sync_sensors_test")
    sensor = SynchronizedSensors(
        color_name="/camera/color",
        depth_name="/camera/aligned_depth_to_color",
        scan_topic="/scan",
        pose_topic="state_estimator/pose_filtered",
    )
    rate = rospy.Rate(10)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        t1 = rospy.Time.now()
        print((t1 - t0).to_sec())
        times = sensor.get_times()
        for k, v in times.items():
            print("-", k, v - t0.to_sec())
        rate.sleep()
