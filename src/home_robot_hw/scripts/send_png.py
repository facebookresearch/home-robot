# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

from home_robot.hw.ros.image_transport import ImageClient

if __name__ == "__main__":
    rospy.init_node("image_client")
    client = ImageClient(show_sizes=False)
    client.spin(rate=15)
