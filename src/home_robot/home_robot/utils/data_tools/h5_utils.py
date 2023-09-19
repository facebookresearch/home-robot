# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import h5py
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from matplotlib import pyplot as plt
from tf2_ros import tf2_ros

from home_robot.utils.data_tools.image import img_from_bytes


def view_keyframe_imgs(file_object: h5py.File, trial_name: str):
    """utility to view keyframe images for named trial from h5 file"""
    num_keyframes = len(file_object[f"{trial_name}/head_rgb"].keys())
    for i in range(num_keyframes):
        _key = f"{trial_name}/head_rgb/{i}"
        img = img_from_bytes(file_object[_key][()])
        plt.imshow(img)
        plt.show()


def plot_ee_pose(
    file_object: h5py.File, trial_name: str, ros_pub: tf2_ros.TransformBroadcaster
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """plot keyframes as TF to be visualized in RVIZ, also return the ee pose associated with them"""
    num_keyframes = len(file_object[f"{trial_name}/ee_pose"][()])
    ee_pose = []
    for i in range(num_keyframes):
        pos = file_object[f"{trial_name}/ee_pose"][()][i][:3]
        rot = file_object[f"{trial_name}/ee_pose"][()][i][3:]
        ee_pose.append((pos, rot))
        pose_message = TransformStamped()
        pose_message.header.stamp = rospy.Time.now()
        pose_message.header.frame_id = "base_link"

        pose_message.child_frame_id = f"key_frame_{i}"
        pose_message.transform.translation.x = pos[0]
        pose_message.transform.translation.y = pos[1]
        pose_message.transform.translation.z = pos[2]

        pose_message.transform.rotation.x = rot[0]
        pose_message.transform.rotation.y = rot[1]
        pose_message.transform.rotation.z = rot[2]
        pose_message.transform.rotation.w = rot[3]

        ros_pub.sendTransform(pose_message)
        input("Press enter to continue")

    return ee_pose
