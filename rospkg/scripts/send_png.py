import rospy
import imagiz
import cv2
import threading
import numpy as np
from home_robot.hw.ros.camera import RosCamera

rospy.init_node('png_sender')
#client = imagiz.Client("cc1", server_ip="192.168.0.79", server_port=5555)
color_client = imagiz.TCP_Client(client_name="color", server_ip="192.168.0.79", server_port=9990)
depth_client = imagiz.TCP_Client(client_name="depth", server_ip="192.168.0.79", server_port=9991)
clients = [color_client, depth_client]

color_camera = RosCamera("/camera/color")
depth_camera = RosCamera("/camera/aligned_depth_to_color")
cameras = [color_camera, depth_camera]

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
def encode_color(frame):
    r, image = cv2.imencode(".jpg", frame, encode_param)
    return image
def encode_depth(frame):
    r, image = cv2.imencode(".png", frame * 1000)
    return image.astype(np.uint16)

encoders = [encode_color, encode_depth]


print("Waiting for images from ROS...")
rate = rospy.Rate(30)
while not rospy.is_shutdown():
    frame = color_camera.get()
    if frame is not None:
        print("Sending frame...")
        color_client.send(encode_color(frame))
    rate.sleep()

"""
import rospy
import imagiz
import cv2
import threading
import numpy as np
from home_robot.hw.ros.camera import RosCamera
from sensor_msgs.msg import Image


rospy.init_node('png_sender')
#client = imagiz.Client("cc1", server_ip="192.168.0.79", server_port=5555)
color_client = imagiz.TCP_Client(client_name="color", server_ip="192.168.0.79", server_port=9990)
depth_client = imagiz.TCP_Client(client_name="depth", server_ip="192.168.0.79", server_port=9991)
clients = [color_client, depth_client]

color_camera = RosCamera("/camera/color")
depth_camera = RosCamera("/camera/aligned_depth_to_color")
cameras = [color_camera, depth_camera]

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
def encode_color(frame):
    r, image = cv2.imencode(".jpg", frame, encode_param)
    return image
def encode_depth(frame):
    r, image = cv2.imencode(".png", frame * 1000)
    return image.astype(np.uint16)

encoders = [encode_color, encode_depth]

print("Waiting for images from ROS...")
rate = rospy.Rate(15)
while not rospy.is_shutdown():
    for camera, client, encode in zip(cameras, clients, encoders):
        frame = camera.get()
        print(frame)
        if frame is not None:
            r, image = cv2.imencode(".jpg", frame, encode_param)
            #client.send(encode(frame))
            client.send(frame)
        continue
    rate.sleep()
"""
