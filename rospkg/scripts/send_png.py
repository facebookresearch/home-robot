import rospy
import imagiz
import cv2

client = imagiz.Client("cc1", server_ip="192.168.0.79")
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image

sub_color = rospy.Subscriber("/camera/color/image_raw", Image, callback_color)


def callback(msg):
    img = image_to_numpy(msg)


rate = rospy.Rate(30)
while not rospy.is_shutdown():
    frame = None
    with lock:
        if img is not None:
            frame = img.copy()
            img = None
    if frame is not None:
        r, image = cv2.imencode(".jpg", frame, encode_param)
        client.send(image)
    else:
        break
    rate.sleep()
