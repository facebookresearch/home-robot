#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import rospy
from sensor_msgs.msg import Image

from home_robot_hw.ros.msg_numpy import image_to_numpy, numpy_to_image

pub_color = None
pub_rotated_color = None
pub_depth = None
pub_rotated_depth = None


def callback_color(msg):
    img = image_to_numpy(msg)
    img = np.rot90(np.rot90(np.rot90(img)))
    msg2 = numpy_to_image(img, msg.encoding)
    msg2.header = msg.header
    pub_color.publish(msg2)
    pub_rotated_color.publish(msg2)


def callback_depth(msg):
    img = image_to_numpy(msg)
    img = np.rot90(np.rot90(np.rot90(img)))
    msg2 = numpy_to_image(img, msg.encoding)
    msg2.header = msg.header
    pub_depth.publish(msg2)
    pub_rotated_depth.publish(msg2)


if __name__ == "__main__":
    rospy.init_node("rotate_images")
    pub_color = rospy.Publisher("/color/image_raw", Image, queue_size=2)
    pub_depth = rospy.Publisher("/depth/image_raw", Image, queue_size=2)
    pub_rotated_color = rospy.Publisher("/rotated_color", Image, queue_size=2)
    pub_rotated_depth = rospy.Publisher("/rotated_depth", Image, queue_size=2)
    sub_color = rospy.Subscriber("/camera/color/image_raw", Image, callback_color)
    sub_depth = rospy.Subscriber(
        "/camera/aligned_depth_to_color/image_raw", Image, callback_depth
    )
    rospy.spin()
