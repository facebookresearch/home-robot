import rospy
from visualization_msgs.msg import Marker
from home_robot_hw.ros.utils import matrix_to_pose_msg


class Visualizer(object):
    """Simple visualizer to send a single marker message"""

    def __init__(self, topic_name, rgba=None):
        self.pub = rospy.Publisher(topic_name, Marker, queue_size=1)
        if rgba is None:
            rgba = [1, 0, 0, 0.75]
        self.rgba = rgba

    def __call__(self, pose_matrix):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.ARROW
        marker.pose = matrix_to_pose_msg(pose_matrix)
        marker.color.r = self.rgba[0]
        marker.color.g = self.rgba[1]
        marker.color.b = self.rgba[2]
        marker.color.a = self.rgba[3]
        marker.scale.x = 0.2
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        self.pub.publish(marker)
