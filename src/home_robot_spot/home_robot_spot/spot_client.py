# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import timeit
from threading import Event, Thread
from typing import Dict, List, Optional, Sequence, Tuple, Union

import bosdyn.client.frame_helpers as frame_helpers
import cv2
import numpy as np
import torch
import transforms3d as t3d
import trimesh.transformations as tra
from bosdyn.api import image_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs
from spot_wrapper.basic_streaming_visualizer_numpy import obstacle_grid_points
from spot_wrapper.spot import Spot, build_image_request, image_response_to_cv2

from home_robot.core.interfaces import Action, Observations
from home_robot.motion import PlanResult
from home_robot.perception.midas import Midas
from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
from home_robot.utils.config import get_config
from home_robot.utils.geometry import (
    angle_difference,
    sophus2xyt,
    xyt2sophus,
    xyt_base_to_global,
)
from home_robot.utils.image import Camera as PinholeCamera
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

RGB_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
DEPTH_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
BODY_THERSH = 10000 / 255
HAND_THERSH = 6000 / 255
RGB_THRESH = 1
RGB_TO_BGR = [2, 1, 0]
CAMERA_SOURCES = [
    ("hand_depth_in_hand_color_frame", DEPTH_FORMAT, HAND_THERSH, None),
    ("hand_color_image", RGB_FORMAT, RGB_THRESH, None),
]
IGNORED_SOURCES = [
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
]
IGNORED_NAMES = [
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
        self.observations: Dict[str, torch.tensor_split] = {}
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
        self.observation_index = 0

        print("waiting for first observation")
        while not self.updated:
            time.sleep(0.1)
        print("got first observation")

    def stop(self):
        # Close all threads
        for thread in self.threads:
            thread.join()

        # kill threads
        del self.threads

    def update_obs(
        self,
    ):
        last_update = time.time()
        while True:
            self.observations = self.get_cam_observations()
            self.observation_index += 1
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

    def get_cam_observations(self, debug: bool = False):
        if debug:
            response_dict = {}
            for req in self.reqs:
                try:
                    response = self.spot.image_client.get_image([req])[0]
                    response_dict[req.image_source_name] = response
                except Exception as e:
                    print(
                        f"-----------------\n{req=}\nException: {e}\n-----------------"
                    )
            responses = list(response_dict.values())
        else:
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


def get_camera_from_spot_image_response(response: image_pb2.ImageResponse):
    # parent_tform_child	SE3Pose		Transform representing the pose of the child frame in the parent's frame.
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#bosdyn-api-FrameTreeSnapshot
    # Build camera model
    tform_snap = response.shot.transforms_snapshot
    body_tform_camera = frame_helpers.get_a_tform_b(
        tform_snap, frame_helpers.BODY_FRAME_NAME, response.shot.frame_name_image_sensor
    ).to_matrix()
    body_tform_vision = frame_helpers.get_a_tform_b(
        tform_snap, frame_helpers.BODY_FRAME_NAME, frame_helpers.VISION_FRAME_NAME
    ).to_matrix()
    cam_to_world = torch.from_numpy(body_tform_vision).inverse() @ torch.from_numpy(
        body_tform_camera
    )
    height = response.shot.image.rows
    width = response.shot.image.cols
    camera = PinholeCamera(
        pos=cam_to_world[:3, 3],
        orn=cam_to_world[:3, :3],
        height=height,
        width=width,
        fx=response.source.pinhole.intrinsics.focal_length.x,
        fy=response.source.pinhole.intrinsics.focal_length.y,
        px=response.source.pinhole.intrinsics.principal_point.x,
        py=response.source.pinhole.intrinsics.principal_point.y,
        near_val=None,
        far_val=None,
        pose_matrix=None,
        proj_matrix=None,
        view_matrix=None,
        fov=None,
    )

    # # Later on we can try converting between conventions. Here's a non-working example
    # from home_robot.utils.camera_view_conventions import convert_camera_view_coordinate_system, CameraViewCoordSystem
    # SPOT = {'up': '-Y', 'right': '+X', 'forward': '+Z'}
    # M = convert_camera_view_coordinate_system(M, source_convention=SPOT, output_convention=CameraViewCoordSystem.OPENGL.value).transformedRT
    # cams = cameras_from_opencv_projection(
    #     R=M[:, :3, :3], # .permute([0,2,1])
    #     tvec=M[:, :3, 3],
    #     # tvec = torch.stack(tvecs).float(),
    #     camera_matrix=torch.stack(intrinsics).float(),
    #     image_size=torch.stack(image_sizes).float()
    # )
    # # cams = PerspectiveCameras(
    # #     device="cpu",
    # #     R=M[:, :3, :3].permute([0,2,1]),
    # #     T=M[:, :3, 3],
    # #     K=torch.stack(intrinsics).float())
    return camera


def torch_image_from_spot_image_response(
    image_response: image_pb2.ImageResponse, reorient=False
):
    is_depth = (
        image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
    )
    if is_depth:
        dtype = np.uint16
    else:
        dtype = np.uint8
        # scale = None
    # img = np.fromstring(image_response.shot.image.data, dtype=dtype)
    img = np.frombuffer(image_response.shot.image.data, dtype=dtype)

    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
    else:
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED) / 255.0  # -1

    if reorient:
        img = np.rot90(img, k=3)

    if is_depth:
        img = img * 1.0 / image_response.source.depth_scale
    return torch.from_numpy(img).float()


def build_obs_from_spot_image_responses(
    depth_response: image_pb2.ImageResponse,
    color_response: Optional[image_pb2.ImageResponse],
):
    # Get camera parameters
    camera: PinholeCamera = get_camera_from_spot_image_response(color_response)
    cam_to_world = torch.eye(4)
    cam_to_world[:3, :3] = camera.orn
    cam_to_world[:3, 3] = camera.pos

    # get images
    color = None
    if color_response is not None:
        color = torch_image_from_spot_image_response(color_response)
    depth = torch_image_from_spot_image_response(depth_response)

    # Other metadata
    timestamp = depth_response.shot.acquisition_time.ToDatetime()

    # Build obs
    return Observations(
        gps=None,
        compass=None,
        rgb=color,
        depth=depth,
        camera_pose=cam_to_world,
        camera_K=camera.K[:3, :3],
        task_observations={"timestamp": timestamp},
    )


def create_triad_pointclouds(R, T, n_points=1, scale=0.1):
    """This will move to home_robot.utils"""
    batch_size = R.shape[0]
    # Define the coordinates of the triad
    triad_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # Origin
            [1.0, 0.0, 0.0, 1.0],  # X-axis
            [0.0, 1.0, 0.0, 1.0],  # Y-axis
            [0.0, 0.0, 1.0, 1.0],
        ]
    )  # Z-axis
    triad_coords = torch.cat(
        [triad_coords * (scale * 1.0 / n_points * i) for i in range(1, n_points + 1)],
        dim=0,
    )
    triad_coords[:, 3] = 1.0
    M = torch.zeros((batch_size, 4, 4))

    M[:, :3, :3] = R
    M[:, :3, 3] = T
    M[:, -1, -1] = 1.0

    triad_coords = triad_coords.unsqueeze(0).expand(
        batch_size, *triad_coords.shape[-2:]
    )
    triad_coords = torch.bmm(triad_coords, M.permute([0, 2, 1]))
    triad_coords = triad_coords[..., :3]
    # triad_coords += T.unsqueeze(1)

    # Create colors for each point (red, green, blue)
    colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # White
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ]
    )
    colors = (
        colors.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_points, 4, 3)
        .reshape(batch_size, n_points * 4, 3)
    )
    return Pointclouds(points=triad_coords[..., :3], features=colors)


class SpotClient:
    def __init__(
        self,
        config,
        name="home_robot_spot",
        dock_id: Optional[int] = None,
        use_midas=False,
        use_zero_depth=True,
    ):
        self.spot = Spot(name)
        self.lease = None
        self.publishers = SpotPublishers(self.spot, verbose=False)
        self.dock_id = dock_id

        # Parameters from Config
        self.config = config
        self.gaze_arm_joint_angles = np.deg2rad(config.SPOT.GAZE_ARM_JOINT_ANGLES)
        self.place_arm_joint_angles = np.deg2rad(config.SPOT.PLACE_ARM_JOINT_ANGLES)
        self.max_cmd_duration = config.SPOT.MAX_CMD_DURATION
        self.hand_depth_threshold = config.SPOT.HAND_DEPTH_THRESHOLD
        self.base_height = config.SPOT.BASE_HEIGHT
        self.use_midas = use_midas
        self.use_zero_depth = use_zero_depth
        self.midas = None
        if self.use_midas:
            self.midas = Midas("cuda:0")
        if self.use_zero_depth:
            self.zerodepth_model = (
                torch.hub.load(
                    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
                )
                .to("cuda:0")
                .eval()
            )

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

    def patch_depth(self, rgb, depth):
        monocular_estimate, mse, mean_error = self.midas.depth_estimate(rgb, depth)

        # clip at 0 if the linear transformation makes some points negative depth
        monocular_estimate[monocular_estimate < 0] = 0

        # threshold max distance to for estimated depth
        # monocular_estimate[monocular_estimate > self.estimated_depth_threshold] = 0

        try:
            # assign estimated depth where there are no values
            no_depth_mask = depth == 0

            # get a mask for the region of the image which has depth values (skip the blank borders)
            row, cols = np.where(~no_depth_mask)
            col_inds = np.indices(depth.shape)[1]
            depth_region = (col_inds >= cols.min()) & (col_inds <= cols.max())
            no_depth_mask = no_depth_mask & depth_region

            depth[no_depth_mask] = monocular_estimate[no_depth_mask]
            return depth
        except Exception as e:
            print(f"Initializing Midas depth completion failed: {e}")
            return depth

    def reset_arm(self):
        self.spot.set_arm_joint_positions(self.gaze_arm_joint_angles, travel_time=1.0)
        self.spot.open_gripper()

    def reset(self):
        """Put arm back in home position and compute epsiode start pose"""
        self.reset_arm()
        self._episode_start_pose = xyt2sophus([self.gps[0], self.gps[1], self.compass])

    def execute_plan(
        self,
        plan: PlanResult,
        verbose: bool = True,
        pos_err_threshold: float = 0.1,
        rot_err_threshold: float = 0.3,
        per_step_timeout: float = 1.0,
    ):
        """go through a whole plan and execute it"""
        if verbose:
            print("Executing motion plan:")
        for i, node in enumerate(plan.trajectory):
            if verbose:
                print(" - go to", i, "xyt =", node.state)
            self.navigate_to(node.state, blocking=False)
            t0 = timeit.default_timer()
            while timeit.default_timer() - t0 < per_step_timeout:
                # Check error
                pose = self.current_position.copy()
                pos_err = np.linalg.norm(pose[:2] - node.state[:2])
                rot_err = angle_difference(pose[2], node.state[2])
                if verbose:
                    print(f"{i} {pos_err=}, {rot_err=}")
                if pos_err < pos_err_threshold and rot_err < rot_err_threshold:
                    break
                time.sleep(0.01)
            if timeit.default_timer() - t0 > per_step_timeout:
                print(f"WARNING: robot could not reach waypoint {i}: {node.state}")
                return False

        # Send us to the final waypoint, but this time actually block - we want to really get there
        self.navigate_to(node.state, blocking=True)
        return True

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
        quat = self.raw_observations["base_xyt"][0][1]
        rpy = tra.euler_from_matrix(tra.quaternion_matrix(quat))
        return rpy[2]

    @property
    def current_relative_position(self) -> np.ndarray:
        # xy = self.gps
        # compass = self.compass
        current_pose = xyt2sophus([self.gps[0], self.gps[1], self.compass])
        # NOTE: old code
        # relative_gps = xy - self.start_gps
        # relative_gps = self.rot_compass @ relative_gps
        # relative_compass = put_angle_in_interval(compass - self.start_compass)
        # return np.array([relative_gps[0], relative_gps[1], relative_compass])
        relative_pose = self._episode_start_pose.inverse() * current_pose
        return sophus2xyt(relative_pose)

    @property
    def current_position(self):
        xy = self.gps
        compass = self.compass
        return np.array([xy[0], xy[1], compass])

        # # return self.raw_observations["base_xyt"][0][0][:3]
        # x, y, yaw = self.spot.get_xy_yaw()
        # relative_gps = self.gps - self.start_gps
        # relative_gps = self.rot_compass @ relative_gps

        # relative_compass = put_angle_in_interval(self.compass - self.start_compass)
        # return np.array([x, y, relative_compass])

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

    def _get_relative_gps_compass(self):
        """Get gps and compass as separate components from current position relative to start"""
        # Normalize GPS
        xyt = self.current_relative_position.copy()
        relative_gps = xyt[:2]
        relative_compass = np.array([xyt[2]])
        return relative_gps, relative_compass

    def _get_gps_compass(self):
        """Get gps and compass as separate components from current position"""
        return self.gps.copy(), np.array([self.compass])

    @property
    def observations(self):
        """
        Returns the observations from spot on the home-robot format
        """
        obs = self.raw_observations

        # relative_gps, relative_compass = self._get_relative_gps_compass()
        gps, compass = self._get_gps_compass()
        rgb = self.hand_depth

        # Preprocess depth
        depth = self.hand_depth
        depth = depth / 255 * self.hand_depth_threshold
        depth[depth > (self.hand_depth_threshold - 0.05)] = 0

        home_robot_obs = Observations(gps=gps, compass=compass, rgb=rgb, depth=depth)
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
            obs["obstacle_distances"][:, :2] - self._episode_start_pose[:2]
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

    def get_rgbd_obs(self, verbose: bool = False) -> Observations:
        """Get information from the Spot sensors with pose and other information"""
        # Get hand depth and color responses
        # depth_response = self.raw_observations["responses"]["hand_depth_in_hand_color_frame"]
        # color_response = self.raw_observations["responses"]["hand_color_image"]
        depth_response = self.raw_observations["responses"][0]
        color_response = self.raw_observations["responses"][1]
        # Get hand depth and color responses
        # depth_response = self.raw_observations["responses"][4]
        # color_response = self.raw_observations["responses"][5]
        # update observations
        obs = build_obs_from_spot_image_responses(depth_response, color_response)
        obs.rgb = obs.rgb[..., RGB_TO_BGR]
        # keep_mask = (0.4 < obs.depth) & (obs.depth < 4.0)

        # relative_gps, relative_compass = self._get_relative_gps_compass()
        # obs.gps = relative_gps
        # obs.compass = relative_compass
        obs.gps, obs.compass = self._get_gps_compass()
        K = obs.camera_K

        if self.use_midas:
            rgb, depth = obs.rgb, obs.depth
            depth = self.patch_depth(rgb, depth)
            obs.depth = depth
        import torch

        if self.use_zero_depth:

            rgb = obs.rgb.clone().detach()
            # rgb = torch.tensor(obs.rgb)

            # Already in 0,1
            # orig_rgb = rgb / 255.0

            # Take it to RGB
            rgb = rgb[..., [2, 1, 0]]

            # B, C, H, W
            rgb = torch.FloatTensor(rgb[None]).permute(0, 3, 1, 2)
            intrinsics = K[:3, :3]
            intrinsics = torch.FloatTensor(intrinsics[None])

            with torch.no_grad():
                pred_depth = (
                    self.zerodepth_model(rgb.to("cuda:0"), intrinsics.to("cuda:0"))[
                        0, 0
                    ]
                    .detach()
                    .cpu()
                )
            obs.depth = pred_depth
        full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
            depth=obs.depth.unsqueeze(0).unsqueeze(1),
            pose=obs.camera_pose.unsqueeze(0),
            # mask=~keep_mask.unsqueeze(0).unsqueeze(1),
            inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
        )
        obs.xyz = full_world_xyz.view(*list(obs.rgb.shape))
        obs.rgb = obs.rgb * 255
        if verbose:
            print("--- SHAPE INFO ---")
            print("xyz.shape =", obs.xyz.shape)
            print("rgb.shape =", obs.rgb.shape)
            print("depth.shape =", obs.depth.shape)

        # TODO: apply keep mask to instance and semantic classification channels in the observations as well

        return obs

    def make_3d_viz(self, viz_data):
        xyz = viz_data["xyz"]
        colors = viz_data["colors"]
        depths = viz_data["depths"]
        Rs = viz_data["Rs"]
        tvecs = viz_data["tvecs"]
        intrinsics = viz_data["intrinsics"]

        depth_response = self.raw_observations["responses"][
            4
        ]  # frontleft_depth_in_visual_frame
        color_response = self.raw_observations["responses"][
            5
        ]  # frontleft_fisheye_image

        obs = build_obs_from_spot_image_responses(depth_response, color_response)

        # unproject, to visualize pointcloud, but this could build a map
        depth = obs.depth
        color = obs.rgb
        cam_to_world = obs.camera_pose
        K = obs.camera_K
        keep_mask = (0.4 < depth) & (depth < 4.0)
        full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
            depth=depth.unsqueeze(0).unsqueeze(1),
            pose=cam_to_world.unsqueeze(0),
            mask=~keep_mask.unsqueeze(0).unsqueeze(1),
            inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
        )

        xyz.append(full_world_xyz.view(-1, 3))
        colors.append(color[keep_mask])
        depths.append(obs.depth)
        Rs.append(cam_to_world[:3, :3])
        tvecs.append(cam_to_world[:3, 3])
        intrinsics.append(torch.from_numpy(K[:3, :3]))

        # Pointcloud
        pointclouds = Pointclouds(points=xyz, features=colors)
        triads = create_triad_pointclouds(
            torch.stack(Rs, dim=0), torch.stack(tvecs, dim=0), n_points=20, scale=0.2
        )

        fig = plot_scene_with_bboxes(
            plots={
                "Scene": {
                    "Camera Triads": triads,
                    "points": pointclouds
                    # "Cameras": cams,
                }
            },
            xaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            yaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=20_000,
            height=1000,
            # width=1000,
        )
        fig.show()

        return {
            "xyz": xyz,
            "colors": colors,
            "depths": depths,
            "Rs": Rs,
            "tvecs": tvecs,
            "intrinsics": intrinsics,
        }

    def __del__(self):
        self.stop()
        self.publishers.stop()

    def stop(self):
        """Try to safely stop the robot."""
        if self.dock_id is not None:
            try:
                self.spot.dock(self.dock_id)
            except Exception as e:
                print(e)

        self.spot.power_off()

        # How do we close the lease
        self.lease.__exit__(None, None, None)
        self.lease = None

    def open_gripper(self):
        """Open the gripper"""
        self.spot.open_gripper()

    def unnormalize_gps_compass(self, xyt):
        # gps = xyt[:2]
        # compass = xyt[2]

        # Transpose is inverse because its an orthonormal matrix
        # gps = self.rot_compass.T @ gps + self.start_gps
        # UnNormalize compass
        # compass = put_angle_in_interval(compass + self.pass)
        # return np.array([gps[0], gps[1], compass])
        pose = self._episode_start_pose * xyt2sophus(xyt)
        return sophus2xyt(pose)

    def navigate_to(self, xyt: np.ndarray, relative: bool = False, blocking=False):
        """Move the base to a new position.

        Args:
            xyt_position: A tuple of (x, y, theta) in meters and radians.
        """
        assert self.lease is not None, "Must call start() first."

        # Unnormalize GPS and compass
        if not isinstance(xyt, np.ndarray):
            xyt = np.array(xyt)
        # print("nav to before unnorm", xyt)
        # xyt = self.unnormalize_gps_compass(xyt)
        # print("after =", xyt)

        if relative:
            xyt = xyt_base_to_global(xyt, self.current_position)

        self.spot.set_base_position(
            x_pos=xyt[0],
            y_pos=xyt[1],
            yaw=xyt[2],
            end_time=1000,
            max_fwd_vel=0.5,
            max_hor_vel=0.5,
            blocking=blocking,
        )
        return self.raw_observations

    def move_base(self, lin, ang, scaling_factor=0.5):
        base_action = np.array([lin, 0, ang])
        scaled = np.clip(base_action, -1, 1) * scaling_factor
        self.spot.set_base_velocity(*scaled, self.max_cmd_duration)
        return self.raw_observations


class VoxelMapSubscriber:
    """
    This class is used to update the voxel map with spot observations, runs on a separate thread
    """

    def __init__(self, spot, voxel_map, semantic_sensor):
        self.spot = spot
        self.voxel_map = voxel_map
        self.semantic_sensor = semantic_sensor
        self.current_obs = 0

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self, verbose=False):
        last_update = time.time()
        while True:
            if self.current_obs < self.spot.publishers.observation_index:
                obs = self.spot.get_rgbd_obs()
                obs = self.semantic_sensor.predict(obs)
                self.voxel_map.add_obs(obs, xyz_frame="world")

                # FPS ( tested at ~.8)
                if verbose:
                    print("Added observation to voxel map")
                    print("FPS: ", 1 / (time.time() - last_update))
                last_update = time.time()
                self.current_obs += 1


if __name__ == "__main__":
    try:
        config = get_config("projects/spot/configs/config.yaml")[0]
        spot = SpotClient(config=config)

        from home_robot.agent.ovmm_agent import (
            OvmmPerception,
            build_vocab_from_category_map,
            read_category_map_file,
        )
        from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
        from home_robot.utils.config import load_config

        # TODO move these parameters to config
        voxel_size = 0.05
        voxel_map = SparseVoxelMap(resolution=voxel_size, local_radius=0.1)

        # Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes.
        print("- Loading configuration")
        config = load_config(visualize=False)

        print("- Create and load vocabulary and perception model")
        semantic_sensor = OvmmPerception(config, 0, True, module="detic")
        obj_name_to_id, rec_name_to_id = read_category_map_file(
            config.ENVIRONMENT.category_map_file
        )
        vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
        semantic_sensor.update_vocabulary_list(vocab, 0)
        semantic_sensor.set_vocabulary(0)

        # Turn on the robot using the client above
        spot.start()

        # Start thread to update voxel map
        voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
        voxel_map_subscriber.start()

        linear = input("Input Linear: ")
        angular = input("Input Angular: ")

        viz_data: Dict[str, List] = {
            "xyz": [],
            "colors": [],
            "depths": [],
            "Rs": [],
            "tvecs": [],
            "intrinsics": [],
        }

        action_index, visualization_frequency = 0, 7
        while linear != "" and angular != "":
            try:
                spot.move_base(float(linear), float(angular))
            except Exception:
                print("Error -- try again")

            # obs = spot.get_rgbd_obs()
            # obs = semantic_sensor.predict(obs)
            # voxel_map.add_obs(obs, xyz_frame="world")
            print("added, now display something")
            if action_index % visualization_frequency == 0 and action_index > 0:
                print(
                    "Observations processed for the map so far: ",
                    voxel_map_subscriber.current_obs,
                )
                print("Actions taken so far: ", action_index)
                voxel_map.show(backend="open3d", instances=False)

            # To navigate to an instance
            # instance_id = <set instance id>
            # instance_view_id = <set instance view id> (for now it can be random or the first one)
            # instances = voxel_map.get_instances()
            # instance = instances[instance_id]
            # view = instance.instance_views[instance_view_id]
            # gps, compass = view.gps (or pose?) this wint work is some rough pseudocode
            # position = np.array([gps[0], gps[1], compass[0]])
            # spot.navigate_to(position)

            linear = input("Input Linear: ")
            angular = input("Input Angular: ")
            # viz_data = spot.make_3d_viz(viz_data)
            action_index += 1

    except Exception as e:
        print(e)
        spot.stop()
        raise e

    # finally:
    #     spot.stop()
