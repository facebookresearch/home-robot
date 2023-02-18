# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

from home_robot.hw.ros.image_transport import ImageServer

if __name__ == "__main__":
    rospy.init_node("local_republisher")
    server = ImageServer(show_images=True)
    server.spin(rate=15)
