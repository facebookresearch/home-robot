import rospy
import imagiz
import cv2
from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image
from sensor_msgs.msg import Image

color_server = imagiz.TCP_Server(port=9990)
depth_server = imagiz.TCP_Server(port=9991)
servers = [color_server, depth_server]
for server in servers:
    server.start()
while True:
    for server, name in zip(servers, ["color", "depth"]):
        message = server.receive()
        frame = cv2.imdecode(message.image, 1)
        cv2.imshow(name, frame)
        cv2.waitKey(1)
