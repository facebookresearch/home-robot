#!/usr/bin/env python

import time

import rospy
from rosgraph_msgs.msg import Clock


class MasterClock:
    def __init__(self, spin_rate=10000):
        self.spin_rate = spin_rate
        self._dt = 1.0 / self.spin_rate
        self._pub = rospy.Publisher("/clock", Clock, queue_size=1)
        self._t = 0

    def spin(self):
        while not rospy.is_shutdown():
            self._pub.publish(rospy.Time(self._t))
            time.sleep(self._dt)
            self._t += self._dt


if __name__ == "__main__":
    rospy.init_node("master_clock")
    MasterClock().spin()
