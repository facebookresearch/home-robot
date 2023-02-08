import rospy
from visualization_msgs.msg import Marker
from home_robot_hw.ros.utils import matrix_to_pose_msg


class Visualizer(object):
    """ Simple visualizer to send a single marker message """

    def __init__(self, topic_name, rgb):
        self.pub = rospy.Publisher(topic_name, Marker, queue_size=1)

    def __call__(self, pose_matrix):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.ARROW
        marker.pose = matrix_to_pose_msg(pose_matrix)
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 0.75
        self.pub.publish(marker)
