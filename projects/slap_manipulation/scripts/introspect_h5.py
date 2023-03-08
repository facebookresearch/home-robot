import h5py
import rospy
import tf2_ros
from slap_manipulation.utils.h5_utils import plot_ee_pose
from visualization_msgs.msg import Marker

if __name__ == "__main__":
    rospy.init_node("h5_introspection")
    ros_pub = tf2_ros.TransformBroadcaster()
    file = h5py.File("/home/priparashar/h5_test/stretch_testing/03-08_12-33-33.h5", "r")
    plot_ee_pose(file, "0", ros_pub)
