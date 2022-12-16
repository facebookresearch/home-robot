import rospy
import imagiz
import cv2
import threading
import numpy as np
from home_robot.hw.ros.camera import RosCamera
from home_robot.utils.data_tools.image import img_to_bytes 

# Create setup for clients + cameras
rospy.init_node("png_sender")
# client = imagiz.Client("cc1", server_ip="192.168.0.79", server_port=5555)
color_client = imagiz.TCP_Client(
    client_name="color", server_ip="192.168.0.79", server_port=9990
)
depth_client = imagiz.TCP_Client(
    client_name="depth", server_ip="192.168.0.79", server_port=9991
)
clients = [color_client, depth_client]

color_camera = RosCamera("/camera/color")
depth_camera = RosCamera("/camera/aligned_depth_to_color")
cameras = [color_camera, depth_camera]

# Encode everything
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
show_sizes = False


def encode_color(frame):
    # r, image = cv2.imencode(".jpg", frame, encode_param)
    image = img_to_bytes(frame, format="webp")
    # image = img_to_bytes(frame)
    if show_sizes:
        print("color len =", len(image))
    return image


def encode_depth(frame):
    frame = (frame * 1000).astype(np.uint16)
    # r, image = cv2.imencode(".png", frame)
    image = img_to_bytes(frame, format="png")
    if show_sizes:
        print("depth len =", len(image))
    return image


encoders = [encode_color, encode_depth]

print("Waiting for images from ROS...")
rate = rospy.Rate(15)
while not rospy.is_shutdown():
    for camera, client, encode in zip(cameras, clients, encoders):
        frame = camera.get().copy()
        if frame is not None:
            client.send(encode(frame))
    rate.sleep()
print("Done.")
