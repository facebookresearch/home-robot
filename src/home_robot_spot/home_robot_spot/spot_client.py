# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from threading import Event, Thread

import cv2
import numpy as np
import torch
import transforms3d as t3d
from bosdyn.api import image_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from spot_wrapper.basic_streaming_visualizer_numpy import obstacle_grid_points
from spot_wrapper.spot import Spot, build_image_request, image_response_to_cv2

from home_robot.core.interfaces import Action, Observations
from home_robot.utils.config import get_config

RGB_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
DEPTH_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
BODY_THERSH = 10000 / 255
HAND_THERSH = 6000 / 255
RGB_THRESH = 1
CAMERA_SOURCES = [
    ("hand_depth_in_hand_color_frame", DEPTH_FORMAT, HAND_THERSH, None),
    ("hand_color_image", RGB_FORMAT, RGB_THRESH, None),
    ("back_depth_in_visual_frame", DEPTH_FORMAT, BODY_THERSH, None),
    ("back_fisheye_image", RGB_FORMAT, RGB_THRESH, None),
    (
        "frontleft_depth_in_visual_frame",
        DEPTH_FORMAT,
        BODY_THERSH,
        cv2.ROTATE_90_CLOCKWISE,
    ),
    ("frontleft_fisheye_image", RGB_FORMAT, RGB_THRESH, cv2.ROTATE_90_CLOCKWISE),
    (
        "frontright_depth_in_visual_frame",
        DEPTH_FORMAT,
        BODY_THERSH,
        cv2.ROTATE_90_CLOCKWISE,
    ),
    ("frontright_fisheye_image", RGB_FORMAT, RGB_THRESH, cv2.ROTATE_90_CLOCKWISE),
    ("left_depth_in_visual_frame", DEPTH_FORMAT, BODY_THERSH, None),
    ("left_fisheye_image", RGB_FORMAT, RGB_THRESH, None),
    ("right_depth_in_visual_frame", DEPTH_FORMAT, BODY_THERSH, cv2.ROTATE_180),
    ("right_fisheye_image", RGB_FORMAT, RGB_THRESH, cv2.ROTATE_180),
]

SENSOR_FRAME_NAMES = [
    "hand_color_image_sensor",
    "hand_color_image_sensor",
    "back_fisheye",
    "back_fisheye",
    "frontleft_fisheye",
    "frontleft_fisheye",
    "frontright_fisheye",
    "frontright_fisheye",
    "left_fisheye",
    "left_fisheye",
    "right_fisheye",
    "right_fisheye",
]


class SpotPublishers:
    def __init__(self, spot: Spot, rotate=False, quality_percent=75, verbose=False):
        self.spot = spot
        self.sources = CAMERA_SOURCES
        self.reqs = [
            build_image_request(
                name,
                quality_percent=quality_percent,
                pixel_format=format,
                resize_ratio=1,
            )
            for name, format, _, _ in self.sources
        ]
        self.rotate = rotate
        self.finished_initial_capture = False
        self.verbose = verbose

    def start(self):
        # Should we do one thread per observation? It currently takes
        self.threads = [
            Thread(target=self.update_obs),
            Thread(target=self.update_obstacle_map),
        ]
        for thread in self.threads:
            thread.daemon = True
            thread.start()

        self.updated = False

        print("waiting for first observation")
        while not self.updated:
            time.sleep(0.1)
        print("got first observation")

    def stop(self):
        # Close all threads
        for thread in self.threads:
            thread.join()

    def update_obs(
        self,
    ):
        last_update = time.time()
        while True:
            self.observations = self.get_cam_observations()
            self.observations.update(self.get_obstacle_map())
            self.updated = True
            if self.verbose:
                print("FPS: ", 1 / (time.time() - last_update))
            last_update = time.time()

    def update_obstacle_map(self):
        while True:
            self.observations.update(self.get_obstacle_map())

    def rot2npy(self, rotation):
        return np.array(
            [
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            ]
        )

    def intrins_to_dict(self, intrins):
        return dict(
            fx=intrins.focal_length.x,
            fy=intrins.focal_length.y,
            px=intrins.principal_point.x,
            py=intrins.principal_point.y,
        )

    def get_cam_observations(self):
        responses = self.spot.image_client.get_image(self.reqs)

        snapshots = [res.shot.transforms_snapshot for res in responses]

        transforms = [
            get_a_tform_b(sn, "body", name).to_matrix()
            for sn, name in zip(snapshots, SENSOR_FRAME_NAMES)
        ]

        parent_poses = [
            get_a_tform_b(sn, "body", name)
            for sn, name in zip(snapshots, SENSOR_FRAME_NAMES)
        ]
        poses = [(np.array([p.x, p.y, p.z]), self.rot2npy(p.rot)) for p in parent_poses]
        base_xyts = [get_a_tform_b(sn, "vision", "body") for sn in snapshots]
        base_xyts = [
            (np.array([p.x, p.y, p.z]), self.rot2npy(p.rot)) for p in base_xyts
        ]

        base_poses = []
        for sn, name in zip(snapshots, SENSOR_FRAME_NAMES):
            tform = get_a_tform_b(sn, "vision", "body")
            yaw = math_helpers.quat_to_eulerZYX(tform.rotation)[0]
            base_poses.append(np.array([tform.x, tform.y, yaw]))

        intrinsics = [
            self.intrins_to_dict(x.source.pinhole.intrinsics) for x in responses
        ]
        images = [image_response_to_cv2(x) for x in responses]
        images = [
            np.clip(im / dat[2], 0, 255).astype(np.uint8)
            for im, dat in zip(images, CAMERA_SOURCES)
        ]
        if self.rotate:
            images = [
                cv2.rotate(im, dat[3]) if dat[3] is not None else im
                for im, dat in zip(images, CAMERA_SOURCES)
            ]
        return {
            "base_xyt": base_xyts,
            "base_pose": base_poses,
            "cam_poses": poses,
            "images": images,
            "intrinsics": intrinsics,
            "snapshots": snapshots,
            "transforms": transforms,
            "responses": responses,
        }

    def get_obstacle_map(self):
        proto = self.spot.local_grid_client.get_local_grids(["obstacle_distance"])
        points, dists = obstacle_grid_points(proto)

        # 16384 x 3 matrix first two columns are x and y in the vision frame (world coordinates)
        # last column is distance from nearest objstacle
        res = np.concatenate((points, dists[:, None]), axis=-1)
        return {"obstacle_distances": res}


def yaw_rotation_matrix_2D(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return rotation_matrix


def yaw_rotation_matrix_3D(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array(
        [
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ]
    )
    return rotation_matrix


def put_angle_in_interval(angle):
    angle = np.mod(angle, 2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


class SpotClient:
    def __init__(self, config, name="home_robot_spot"):
        self.spot = Spot(name)
        self.lease = None
        self.publishers = SpotPublishers(self.spot, verbose=False)

        # Parameters from Config
        self.config = config
        self.gaze_arm_joint_angles = np.deg2rad(config.SPOT.GAZE_ARM_JOINT_ANGLES)
        self.place_arm_joint_angles = np.deg2rad(config.SPOT.PLACE_ARM_JOINT_ANGLES)
        self.MAX_CMD_DURATION = config.SPOT.MAX_CMD_DURATION
        self.hand_depth_threshold = config.SPOT.HAND_DEPTH_THRESHOLD
        self.base_height = config.SPOT.BASE_HEIGHT

    def start(self):
        """Turn on the robot, stand up, etc."""

        # Get lease and power on
        self.lease = self.spot.get_lease(hijack=True)
        self.spot.power_on()

        # Undock and start publishers
        try:
            self.spot.undock()
        except Exception:
            self.spot.blocking_stand()
        self.publishers.start()

        # Reset
        self.reset()

    def reset(self):
        self.spot.set_arm_joint_positions(self.gaze_arm_joint_angles, travel_time=1.0)
        self.spot.open_gripper()

        self.start_gps = self.gps
        self.start_compass = self.compass
        self.rot_compass = yaw_rotation_matrix_2D(-self.start_compass)

    @property
    def raw_observations(self):
        """
        Return the raw observations from spot
        """
        return self.publishers.observations

    @property
    def gps(self):
        return self.raw_observations["base_xyt"][0][0][:2]

    @property
    def compass(self):
        return self.raw_observations["base_xyt"][0][0][2]

    @property
    def hand_depth(self):
        return self.raw_observations["images"][0]

    @property
    def hand_rgb(self):
        return self.raw_observations["images"][1]

    @property
    def hand_camera_position(self):
        return self.raw_observations["cam_poses"][0][0]

    @property
    def hand_camera_rotation(self):
        return self.raw_observations["cam_poses"][0][1]

    @property
    def observations(self):
        """
        Returns the observations from spot on the home-robot format
        """
        obs = self.raw_observations

        # Normalize GPS
        relative_gps = self.gps - self.start_gps
        relative_gps = self.rot_compass @ relative_gps

        # Normalize compass
        relative_compass = np.array(
            [put_angle_in_interval(self.compass - self.start_compass)]
        )

        rgb = self.hand_depth

        # Preprocess depth
        depth = self.hand_depth
        depth = depth / 255 * self.hand_depth_threshold
        depth[depth > (self.hand_depth_threshold - 0.05)] = 0

        home_robot_obs = Observations(
            gps=relative_gps, compass=relative_compass, rgb=rgb, depth=depth
        )
        home_robot_obs.raw_obs = obs

        # Camera pose
        home_robot_obs.camera_pose = np.zeros((4, 4))
        home_robot_obs.camera_pose[:2, 3] = (
            self.hand_camera_position[:2] + home_robot_obs.gps
        )
        home_robot_obs.camera_pose[2, 3] = (
            self.hand_camera_position[2] + self.base_height
        )
        rel_rot = t3d.quaternions.quat2mat(self.hand_camera_rotation)
        abs_rot = yaw_rotation_matrix_3D(home_robot_obs.compass[0]) @ rel_rot
        home_robot_obs.camera_pose[:3, :3] = abs_rot

        relative_obs_locations = (
            obs["obstacle_distances"][:, :2] - self.start_gps
        ).copy()
        relative_obs_locations = (self.rot_compass @ relative_obs_locations.T).T[
            :, ::-1
        ]

        trusted_point = (
            np.linalg.norm(obs["obstacle_distances"][:, :2] - self.gps, axis=-1) <= 1.5
        )
        obstacle_threshold = 0.01
        is_obstacle_mask = obs["obstacle_distances"][:, 2] <= obstacle_threshold
        is_free_mask = obs["obstacle_distances"][:, 2] > obstacle_threshold

        home_robot_obs.task_observations = {}
        home_robot_obs.task_observations["obstacle_locations"] = torch.from_numpy(
            relative_obs_locations[is_obstacle_mask & trusted_point]
        )
        home_robot_obs.task_observations["free_locations"] = torch.from_numpy(
            relative_obs_locations[is_free_mask & trusted_point]
        )
        return home_robot_obs

    def __del__(self):
        self.stop()
        self.publishers.stop()

    def stop(self):
        try:
            self.spot.dock()
        except Exception as e:
            print(e)

        self.spot.power_off()
        self.lease.close()
        self.lease = None

    def move_base_point(self, xyt: np.ndarray):
        """Move the base to a new position.

        Args:
            xyt_position: A tuple of (x, y, theta) in meters and radians.
        """
        assert self.lease is not None, "Must call start() first."
        self.spot.set_base_position(
            x_pos=xyt[0], y_pos=xyt[1], yaw=xyt[2], end_time=100
        )
        return self.raw_observations

    def move_base(self, lin, ang, scaling_factor=0.5):
        base_action = np.array([lin, 0, ang])
        scaled = np.clip(base_action, -1, 1) * scaling_factor
        self.spot.set_base_velocity(*scaled, self.MAX_CMD_DURATION)

        return self.raw_observations


if __name__ == "__main__":
    try:
        config = get_config("projects/spot/configs/config.yaml")[0]
        spot = SpotClient(config=config)
        spot.start()
        print("Got all images")

        # spot.move_base(0.5, 0.0)
        breakpoint()

    except Exception as e:
        print(e)
        spot.stop()
        raise e

    # finally:
    #     spot.stop()
