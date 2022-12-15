import rospy
import imagiz
import cv2
from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image
from sensor_msgs.msg import Image

rospy.init_node("local_republisher")

color_server = imagiz.TCP_Server(port=9990)
depth_server = imagiz.TCP_Server(port=9991)

pub_color = rospy.Publisher("/server/color/image_raw", Image, queue_size=2)
pub_depth = rospy.Publisher("/server/depth/image_raw", Image, queue_size=2)
pub_rotated_color = rospy.Publisher("/server/rotated_color", Image, queue_size=2)
pub_rotated_depth = rospy.Publisher("/server/rotated_depth", Image, queue_size=2)


servers = [color_server, depth_server]
for server in servers:
    server.start()
rate = rospy.Rate(15)
while not rospy.is_shutdown():
    for server, name in zip(servers, ["color", "depth"]):
        message = server.receive()
        frame = cv2.imdecode(message.image, 1)
        if name == "color":
            pub_color.publish(numpy_to_image(frame, "8UC3"))
            # cv2.imshow(name, frame)
            # cv2.waitKey(1)
    rate.sleep()
