from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
import transforms3d as t3d
from spot_rl.envs.semnav_env import SpotSemanticNavEnv
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)

from home_robot.core.abstract_env import Env
from home_robot.core.interfaces import Action, Observations

HAND_DEPTH_THRESHOLD = 6.0  # in meters
BASE_HEIGHT = 0.61  # in meters


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


class SpotEnv(Env):
    def __init__(self, spot):
        config = construct_config()
        self.env = SpotSemanticNavEnv(config, spot)
        self.start_gps = None
        self.start_compass = None
        self.rot_compass = None

    # world_pos should be (n x 2) matrix of points in the spot world frame
    def spot_world_to_boot_world(self, world_pos):
        relative_obs_locations = world_pos[:, :2] - self.start_gps
        relative_obs_locations = (self.rot_compass @ relative_obs_locations.T).T
        return relative_obs_locations

    def reset(self):
        self.env.reset()
        observations = self.env.get_observations()
        self.start_gps = observations["position"]
        self.start_compass = observations["yaw"]
        self.rot_compass = yaw_rotation_matrix_2D(-self.start_compass)

    @abstractmethod
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        pass

    def get_observation(self) -> Observations:
        """
        Preprocess Spot observation into home_robot.Observations schema.
        """
        obs = self.env.get_observations()

        # Normalize GPS
        relative_gps = obs["position"] - self.start_gps
        relative_gps = self.rot_compass @ relative_gps

        # Normalize compass
        relative_compass = np.array(
            [put_angle_in_interval(obs["yaw"] - self.start_compass)]
        )

        rgb = obs["hand_rgb"]

        # Preprocess depth
        depth = obs["hand_depth_raw"]
        depth = depth / 255 * HAND_DEPTH_THRESHOLD
        depth[depth > (HAND_DEPTH_THRESHOLD - 0.05)] = 0

        home_robot_obs = Observations(
            gps=relative_gps, compass=relative_compass, rgb=rgb, depth=depth
        )
        home_robot_obs.raw_obs = obs

        # Camera pose
        home_robot_obs.camera_pose = np.zeros((4, 4))
        home_robot_obs.camera_pose[:2, 3] = (
            obs["camera_position"][:2] + home_robot_obs.gps
        )
        home_robot_obs.camera_pose[2, 3] = obs["camera_position"][2] + BASE_HEIGHT
        rel_rot = t3d.quaternions.quat2mat(obs["camera_rotation"])
        abs_rot = yaw_rotation_matrix_3D(home_robot_obs.compass[0]) @ rel_rot
        home_robot_obs.camera_pose[:3, :3] = abs_rot

        relative_obs_locations = (
            obs["obstacle_distances"][:, :2] - self.start_gps
        ).copy()
        relative_obs_locations = (self.rot_compass @ relative_obs_locations.T).T[
            :, ::-1
        ]

        trusted_point = (
            np.linalg.norm(obs["obstacle_distances"][:, :2] - obs["position"], axis=-1)
            <= 1.5
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

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass
