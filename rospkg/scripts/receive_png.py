import rospy
import imagiz
import cv2
import numpy as np
from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image
from home_robot.utils.data_tools.image import img_from_bytes
from sensor_msgs.msg import Image

rospy.init_node("local_republisher")
show_images = True

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
        if name == "color":
            frame = img_from_bytes(message.image, format="webp")
            # frame = cv2.imdecode(message.image, 1)
            pub_color.publish(numpy_to_image(frame, "8UC3"))
        elif name == "depth":
            # frame = cv2.imdecode(message.image, 0)
            frame = (img_from_bytes(message.image, format="png") / 1000.0).astype(
                np.float32
            )
            # print(frame)
            # print(frame.shape)
            # pub_depth.publish(
            #    numpy_to_image((frame / 1000.0).astype(np.float32), "32FC1")
            # )
        if show_images:
            print(frame)
            cv2.imshow(name, frame)
            cv2.waitKey(1)
    rate.sleep()
