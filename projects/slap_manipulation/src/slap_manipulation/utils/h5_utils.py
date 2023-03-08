import h5py
import rospy
from geometry_msgs.msg import TransformStamped
from matplotlib import pyplot as plt
from tf2_ros import tf2_ros

from home_robot.utils.data_tools.image import img_from_bytes


def view_keyframe_imgs(file_object: h5py.File, trial_name: str):
    num_keyframes = len(file_object[f"{trial_name}/rgb"].keys())
    for i in range(num_keyframes):
        _key = f"{trial_name}/rgb/{i}"
        img = img_from_bytes(file_object[_key][()])
        plt.imshow(img)
        plt.show()


def plot_ee_pose(
    file_object: h5py.File, trial_name: str, ros_pub: tf2_ros.TransformBroadcaster
):
    num_keyframes = len(file_object[f"{trial_name}/ee_pose"][()])
    for i in range(num_keyframes):
        pos = file_object[f"{trial_name}/ee_pose"][()][i][:3]
        rot = file_object[f"{trial_name}/ee_pose"][()][i][3:]
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
