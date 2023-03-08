import h5py
import rospy
from matplotlib import pyplot as plt
from visualization_msgs.msg import Marker

from home_robot.utils.data_tools.image import img_from_bytes


def view_keyframe_imgs(file_object: h5py.File, trial_name: str):
    num_keyframes = len(file_object[f"{trial_name}/rgb"].keys())
    for i in range(num_keyframes):
        _key = f"{trial_name}/rgb/{i}"
        img = img_from_bytes(file_object[_key][()])
        plt.imshow(img)
        plt.show()


def plot_ee_pose(file_object: h5py.File, trial_name: str, marker_pub: rospy.Publisher):
    num_keyframes = len(file_object[f"{trial_name}/ee_pose"][()])
    breakpoint()
    for i in range(num_keyframes):
        ee_marker = Marker()
        ee_marker.header.frame_id = "base_link"
        ee_marker.header.stamp = rospy.Time.now()
        ee_marker.ns = "end_effector"
        ee_marker.id = 0
        ee_marker.type = Marker.ARROW
        ee_marker.action = Marker.ADD
        ee_marker.pose.position.x = file_object[f"{trial_name}/ee_pose"][()][i][0]
        ee_marker.pose.position.y = file_object[f"{trial_name}/ee_pose"][()][i][1]
        ee_marker.pose.position.z = file_object[f"{trial_name}/ee_pose"][()][i][2]
        ee_marker.pose.orientation.x = file_object[f"{trial_name}/ee_pose"][()][i][3]
        ee_marker.pose.orientation.y = file_object[f"{trial_name}/ee_pose"][()][i][4]
        ee_marker.pose.orientation.z = file_object[f"{trial_name}/ee_pose"][()][i][5]
        ee_marker.pose.orientation.w = file_object[f"{trial_name}/ee_pose"][()][i][6]
        ee_marker.scale.x = 0.20
        ee_marker.scale.y = 0.05
        ee_marker.scale.z = 0.05
        ee_marker.color.a = 1.0
        ee_marker.color.r = 1.0
        ee_marker.color.g = 0.0
        ee_marker.color.b = 0.0
        marker_pub.publish(ee_marker)
        input("Press enter to continue")
