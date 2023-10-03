#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import rospy
from rosgraph_msgs.msg import Clock


class MasterClock:
    def __init__(self, spin_rate=10000, wall_time=True):
        self.spin_rate = spin_rate
        self.wall_time = wall_time
        self._dt = 1.0 / self.spin_rate
        self._pub = rospy.Publisher("/clock", Clock, queue_size=1)
        self._t = 0

    def spin(self):
        """Spin in a loop while publishing time signal. Because this is the "master" clock used for ROS time thoughout HomeRobot, it must not use ROS sleeps!"""
        while not rospy.is_shutdown():
            if self.wall_time:
                self._pub.publish(rospy.Time(time.time()))
            else:
                self._pub.publish(rospy.Time(self._t))
            time.sleep(self._dt)
            self._t += self._dt


if __name__ == "__main__":
    rospy.init_node("master_clock")
    MasterClock().spin()
