# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import cv2
import h5py
import numpy as np
import rospy
from home_robot.utils.data_tools.image import img_from_bytes
from home_robot.utils.data_tools.writer import DataWriter
from tqdm import tqdm

from home_robot_hw.ros.stretch_ros import HelloStretchROSInterface


class Recorder(object):
    """ROS object that subscribes from information from the robot and publishes it out."""

    def __init__(self, filename, start_recording=False, model=None, robot=None):
        """Collect information"""
        print("Connecting to robot environment...")
        self.robot = (
            HelloStretchROSInterface(visualize_planner=False, model=model)
            if robot is None
            else robot
        )
        print("... done connecting to robot environment")
        self.rgb_cam = self.robot.rgb_cam
        self.dpt_cam = self.robot.dpt_cam
        self.writer = DataWriter(filename)
        self.idx = 0
        self._recording_started = start_recording
        self._filename = filename

    def start_recording(self, task_name):
        self._recording_started = True
        # add camera-info as config
        color_camera_info = self._construct_camera_info(self.robot.rgb_cam)
        depth_camera_info = self._construct_camera_info(self.robot.dpt_cam)
        self.writer.add_config(
            color_camera_info=color_camera_info, depth_camera_info=depth_camera_info, task_name=task_name,
        )

        print(f"Ready to record demonstration to file: {self._filename}")

    def finish_recording(self):
        print(f"... done recording trial named: {self.idx}.")
        self.writer.write_trial(self.idx)
        self.idx += 1

    def _construct_camera_info(self, camera):
        return {
            "distortion_model": camera.distortion_model,
            "D": camera.D,
            "K": camera.K,
            "R": camera.R,
            "P": camera.P,
        }

    def save_frame(self):
        """saves the following to an H5 file:
        1. rgb image
        2. depth image
        3. base pose
        4. joint positions
        5. joint velocities
        6. camera pose
        7. camera info
        8. end-effector pose
        """
        # record rgb and depth
        rgb, depth = self.robot.get_images(compute_xyz=False)
        q, dq = self.robot.update()
        # TODO get the following from TF lookup
        ee_pose = self.robot.model.fk(q)
        base_pose = self.robot.get_base_pose()
        camera_pose = self.robot.get_camera_pose()
        self.writer.add_img_frame(rgb=rgb, depth=(depth * 10000).astype(np.uint16))
        self.writer.add_frame(
            q=q,
            dq=dq,
            ee_pose=ee_pose,
            base_pose=base_pose,
            camera_pose=camera_pose,
        )

        return rgb, depth, q, dq

    def spin(self, rate=10):
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if not self._recording_started:
                print("...")
                rate.sleep()
                continue
            self.save_frame()
            rate.sleep()
        self.finish_recording()

    def close(self):
        """clean-up: delete self"""
        del self


def png_to_mp4(group: h5py.Group, key: str, name: str, fps=10):
    """
    Write key out as a gif
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    writer = None

    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):
        bindata = img_stream[k][()]
        _img = img_from_bytes(bindata)
        w, h = _img.shape[:2]
        img = np.zeros_like(_img)
        img[:, :, 0] = _img[:, :, 2]
        img[:, :, 1] = _img[:, :, 1]
        img[:, :, 2] = _img[:, :, 0]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            writer = cv2.VideoWriter(name, fourcc, fps, (h, w))
        writer.write(img)
    writer.release()


def pngs_to_mp4(filename: str, key: str, vid_name: str, fps: int):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_mp4(group, key, str(vid_name) + "_" + group_name + ".mp4", fps=fps)


def parse_args():
    parser = argparse.ArgumentParser("data recorder v1")
    parser.add_argument("--filename", "-f", default="test-data.h5")
    parser.add_argument("--video", "-v", default="")
    parser.add_argument("--fps", "-r", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    rospy.init_node("record_data_from_stretch")
    print("ready...")
    args = parse_args()
    print(args)
    rec = Recorder(args.filename)
    rec.spin(rate=args.fps)
    if len(args.video) > 0:
        print("Writing video...")
        # Write to video
        pngs_to_mp4(args.filename, "rgb", args.video, fps=args.fps)
