from abc import abstractmethod
import cv2
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
import skfmm

HAND_DEPTH_THRESHOLD = 6.0  # in meters
BASE_HEIGHT = 0.61  # in meters


def fmm_distance(obstacles,source):
    map_obs = np.ones_like(obstacles)
    map_obs[source[0],source[1]] = 0
    marr = np.ma.MaskedArray(map_obs,obstacles)
    dists = skfmm.distance(marr)
    return dists


# returns a mask dicating which regions are occoluded by obstacles from the point
def ray_trace(obstacles,point):
    raw_dists = fmm_distance(np.zeros_like(obstacles),point)
    occ_dists = fmm_distance(obstacles,point)
    occluded_mask = np.abs(occ_dists - raw_dists) > 0.5
    if not isinstance(occluded_mask,np.ma.MaskedArray):
        occluded_mask = np.ma.MaskedArray(occluded_mask,np.zeros_like(occluded_mask))
    return occluded_mask

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
    def spot_world_to_boot_world(self,world_pos):
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

        
        relative_obs_locations = (obs["obstacle_distances"][:, :2] - self.start_gps).copy()
        relative_obs_locations = (self.rot_compass @ relative_obs_locations.T).T[ :, ::-1 ]

        trust_region = 1.3
        trusted_point = (
            np.linalg.norm(obs["obstacle_distances"][:, :2] - obs["position"], axis=-1)
            <= trust_region
        )
        obstacle_threshold = 0.01
        is_obstacle_mask = obs["obstacle_distances"][:, 2] <= obstacle_threshold
        is_free_mask = obs["obstacle_distances"][:, 2] > obstacle_threshold

        home_robot_obs.task_observations = {}
        home_robot_obs.task_observations["obstacle_locations"] = torch.from_numpy(
            relative_obs_locations[is_obstacle_mask & trusted_point]
        )
        free_locations = relative_obs_locations[is_free_mask & trusted_point]
        ray_tracing = True
        if ray_tracing:
            # import pdb; pdb.set_trace()
            # from matplotlib import pyplot as plt 
            # plt.scatter(free_points[:,0],free_points[:,1])
            # plt.show()
            # from IPython import embed; embed()
            # import dill
            # import pickle
            # loc = locals()
            # save_dir = {k: v for k,v in loc.items() if test_pickle(v)}
            # def test_pickle(o):
                # import pickle
                # try:
                    # pickle.dumps(o)
                # except (pickle.PicklingError, TypeError, AttributeError):
                    # return False
                # return True
            # pickle.dump(save_dir, open(f'locals.pkl', 'wb'))

           # obs['obstacle_distances'][:,:2] - obs['position']
            # meters to pixel
            resolution = 0.05
            # pixel_locations = ((obs["obstacle_distances"][:, :2] - obs["position"]) / resolution)
            # print(pixel_locations.min(),pixel_locations.max())
            rad = 3/resolution
            dia = int(2*rad)
            map_region = np.zeros((dia, dia))
            pixel_locations = ((obs["obstacle_distances"][:, :2] - obs["position"]) / resolution) + dia/2
            pixel_locations = pixel_locations[is_obstacle_mask].astype(int)
            map_region[pixel_locations[:, 0], pixel_locations[:, 1]] = 1
            occluded_mask = ray_trace(map_region,(dia//2,dia//2))
            observed_free = (map_region == 0) & ~occluded_mask
            # you have to and with the mask channel, np.where doesn't respect the masked array
            free_points = np.stack(np.where(observed_free & ~observed_free.mask),axis=1)
            free_points = (free_points - dia/2) * resolution
            trusted_free = free_points[np.linalg.norm(free_points, axis=-1) <= trust_region].copy()

            trusted_free += obs['position'] - self.start_gps
            trusted_free = (self.rot_compass @ trusted_free.T).T[ :, ::-1 ].copy()

            # from matplotlib.figure import Figure
            # from matplotlib.backends.backend_agg import FigureCanvasAgg
            # fig = Figure(figsize=(5,4),dpi=100)
            # ax = fig.add_subplot()
            # ax.scatter(trusted_free[:,0],trusted_free[:,1])
            # ca = FigureCanvasAgg(fig)
            # ca.draw()
            # shape,(w,h) = ca.print_to_buffer()
            # img = np.frombuffer(shape,dtype='uint8').reshape((h,w,4))

            # fig = Figure(figsize=(5,4),dpi=100)
            # ax = fig.add_subplot()
            # ax.scatter(free_locations[:,0],free_locations[:,1])
            # ca = FigureCanvasAgg(fig)
            # ca.draw()
            # shape,(w,h) = ca.print_to_buffer()
            # img2 = np.frombuffer(shape,dtype='uint8').reshape((h,w,4))
            # vis = np.concatenate((img,img2),axis=1)
            # cv2.imshow('scatters',vis)
            # print(obs['position'])
            # print(self.start_gps)
            # from matplotlib import pyplot as plt 

            home_robot_obs.task_observations["free_locations"] = torch.from_numpy(
                trusted_free
            )
        else:
            home_robot_obs.task_observations["free_locations"] = torch.from_numpy(
                free_locations
            )
        return home_robot_obs

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass
