import rospy
import imagiz
import cv2
import numpy as np
from home_robot.hw.ros.msg_numpy import image_to_numpy, numpy_to_image
from home_robot.utils.data_tools.image import img_from_bytes
from sensor_msgs.msg import Image, CameraInfo


class ImageServer(object):
    def __init__(self, show_images=False):
        self.show_images = show_images
        self.color_server = imagiz.TCP_Server(port=9990)
        self.depth_server = imagiz.TCP_Server(port=9991)

        self.pub_color = rospy.Publisher("/server/color/image_raw", Image, queue_size=2)
        self.pub_depth = rospy.Publisher("/server/depth/image_raw", Image, queue_size=2)

        # TODO: add rotated images if necessary
        # self.pub_rotated_color = rospy.Publisher("/server/rotated_color", Image, queue_size=2)
        # self.pub_rotated_depth = rospy.Publisher("/server/rotated_depth", Image, queue_size=2)

        # Create camera info publishers
        pub_color_cam_info = rospy.Publisher(
            "/server/color/camera_info", CameraInfo, queue_size=1
        )
        pub_depth_cam_info = rospy.Publisher(
            "/server/depth/camera_info", CameraInfo, queue_size=1
        )

        self.reference_frame = None
        self.sequence_id = 0

        cb_color_cam_info = lambda msg: self.send_and_receive_camera_info(
            msg, pub_color_cam_info
        )
        cb_depth_cam_info = lambda msg: self.send_and_receive_camera_info(
            msg, pub_depth_cam_info
        )
        self.sub_color_cam_info = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, cb_color_cam_info
        )
        self.sub_depth_cam_info = rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo, cb_color_cam_info
        )

    def send_and_receive_camera_info(self, cam_info, publisher):
        if self.reference_frame is None:
            self.reference_frame = cam_info.header.frame_id
        cam_info.header.stamp = rospy.Time.now()
        cam_info.header.seq = self.sequence_id
        publisher.publish(cam_info)

    def spin(self, rate=15):
        servers = [self.color_server, self.depth_server]
        for server in servers:
            server.start()
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            for server, name in zip(servers, ["color", "depth"]):
                message = server.receive()
                if name == "color":
                    frame = img_from_bytes(message.image, format="webp")
                    # frame = cv2.imdecode(message.image, 1)
                    msg = numpy_to_image(frame, "8UC3")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = self.reference_frame
                    msg.header.seq = self.sequence_id
                    self.pub_color.publish(msg)
                elif name == "depth":
                    # frame = cv2.imdecode(message.image, 0)
                    frame = (
                        img_from_bytes(message.image, format="png") / 1000.0
                    ).astype(np.float32)
                    # Publish images to the new topic
                    msg = numpy_to_image((frame / 1000.0).astype(np.float32), "32FC1")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = self.reference_frame
                    msg.header.seq = self.sequence_id
                    self.pub_depth.publish(msg)
                if self.show_images:
                    cv2.imshow(name, frame)
                    cv2.waitKey(1)
            rate.sleep()
            self.sequence_id += 1
        print("Done.")
