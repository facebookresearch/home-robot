# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import math
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import click
import numpy as np
import open3d as o3d
import torch
import trimesh
import trimesh.transformations as tra
import yaml
from slap_manipulation.dataloaders.annotations import load_annotations_dict
from slap_manipulation.dataloaders.rlbench_loader import RLBenchDataset
from slap_manipulation.utils.pointcloud_preprocessing import find_closest_point_to_line

from home_robot.core.interfaces import Observations

# TODO Replace with Stretch embodiment
from home_robot.motion.franka import FrankaPanda
from home_robot.motion.stretch import STRETCH_TO_GRASP
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.data_tools.camera import Camera
from home_robot.utils.data_tools.loader import Trial
from home_robot.utils.image import rotate_image
from home_robot.utils.point_cloud import (
    add_additive_noise_to_xyz,
    depth_to_xyz,
    dropout_random_ellipses,
    numpy_to_pcd,
    show_point_cloud,
)

REAL_WORLD_CATEGORIES = [
    "cup",
    "bottle",
    "drawer",
    "basket",
    "bowl",
]
VOXEL_SIZE_1 = 0.001
VOXEL_SIZE_2 = 0.01
DATA_FACTOR = 1


def show_point_cloud_with_keypt_and_closest_pt(
    xyz: np.ndarray,
    rgb: np.ndarray,
    keyframe_orig: np.ndarray,
    keyframe_rot: np.ndarray,
    closest_pt: np.ndarray,
):
    """
    Method to visualize input point-cloud along with ee pose and labeled interaction point
    Args:
        xyz: (Nx3) point cloud points
        rgb: (Nx3) point cloud color
        keyframe_orig: (3x1 vector) ee/keyframe position
        keyframe_rot: (3x3 matrix) ee/keyframe orientation as rotation matrix
        closest_pt: (3x1 vector) labeled interaction point
    """
    if np.any(rgb) > 1:
        rgb = rgb / np.max(rgb)
    pcd = numpy_to_pcd(xyz, rgb)
    geoms = [pcd]
    if keyframe_orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=keyframe_orig
        )
        if keyframe_rot is not None:
            coords = coords.rotate(keyframe_rot)
        geoms.append(coords)
    if closest_pt is not None:
        closest_pt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        closest_pt_sphere.translate(closest_pt)
        closest_pt_sphere.paint_uniform_color([1, 0.706, 0])
        geoms.append(closest_pt_sphere)
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        )
    )
    o3d.visualization.draw_geometries(geoms)


class RPHighLevelTrial(Trial):
    """handle a domain-randomized trial"""

    def __init__(self, name, h5_filename, dataset, group, factor=DATA_FACTOR):
        """
        Use group for initialization
        """
        super().__init__(name, h5_filename, dataset, group)
        keypoint_array = self["user_keyframe"][()].squeeze()
        idx = np.arange(keypoint_array.shape[0])
        idx = idx[keypoint_array == 1]
        keypoint_len = len(idx)
        # extra samples for metrics - used to coer for randomness in ptnet ops?
        self.factor = factor
        # extra training time spent on dr examples
        self.dr_factor = 5
        self.length = (
            keypoint_len
            * (self.dr_factor if dataset.data_augmentation else 1)
            * self.factor
        )
        self.num_keypoints = keypoint_len
        self.keypoints = idx


class RobotDataset(RLBenchDataset):
    """train on a dataset from robot dataset"""

    def __init__(
        self,
        dirname,
        template="*.h5",
        verbose=False,
        num_pts=8000,
        data_augmentation=True,
        random_cmd=True,
        first_keypoint_only=False,
        keypoint_range: Optional[List] = None,
        show_voxelized_input_and_reference=False,
        show_raw_input_and_reference=False,
        show_cropped=False,
        ori_dr_range=np.pi / 4,
        cart_dr_range=1.0,
        first_frame_as_input=False,
        trial_list: list = [],
        orientation_type="quaternion",
        multi_step=False,
        crop_radius=True,
        ambiguous_radius=0.03,
        crop_radius_chance=0.75,
        crop_radius_shift=0.1,
        crop_radius_range=[0.3, 1.0],
        visualize_interaction_estimates=False,
        visualize_cropped_keyframes=False,
        yaml_file=None,  # "./assets/language_variations/v0.yml",
        robot="stretch",
        depth_factor=10000,
        autoregressive=False,
        max_keypoints=6,
        time_as_one_hot=False,
        per_action_cmd=False,
        skill_to_action_file=None,
        *args,
        **kwargs,
    ):
        """
        dirname:                name of dir with all h5 files
        template:               template for glob to find all h5 files
        verbose:                prints info about number of files, their names and trial info
        num_pts:                number of points to sample from point cloud
        data_augmentation:      whether to apply domain randomization
        random_cmd:             whether to randomly sample a task language variation from the list of commands
        first_keypoint_only:    whether to only use the first keypoint in the sequence
        keypoint_range:         list of keypoint indices to use for training
        show_voxelized_input_and_reference: whether to show voxelized input and reference point clouds
        show_raw_input_and_reference: whether to show raw input and reference point clouds
        show_cropped:           whether to show cropped input point-cloud
        ori_dr_range:           magnitude for domain randomization for orientation
        cart_dr_range:          magnitude for domain randomization for cartesian position
        first_frame_as_input:   whether to only use the first frame in the sequence for input PCD
        trial_list:             list of trials to sample from; if not provided all trials found are used
        orientation_type:       type of orientation to use for training; can be "quaternion" or "euler"
        multi_step:             whether to return output signals for multi-step regression training
        crop_radius:            whether to crop the input point cloud to a sphere of radius crop_radius_range
        robot:                  name of robot (stretch/franka)
        per_action_cmd:         use different language per waypoint
        """
        if yaml_file is not None:
            self.annotations = load_annotations_dict(yaml_file)
        else:
            self.annotations = None
        if skill_to_action_file is not None:
            self.skill_to_action = yaml.load(
                open(skill_to_action_file, "r"), Loader=yaml.FullLoader
            )
        else:
            self.skill_to_action = None
        self.max_keypoints = max_keypoints
        self.time_as_one_hot = time_as_one_hot
        self.per_action_cmd = per_action_cmd
        self.random_cmd = random_cmd
        self.autoregressive = autoregressive
        # TODO: deprecate this and use only keypoint_range to constrain index of sampled keyframe
        self.first_keypoint_only = first_keypoint_only
        self.keypoint_to_use = None
        self.multi_step = multi_step
        self.num_pts = num_pts
        self.data_augmentation = data_augmentation
        self.ori_dr_range = ori_dr_range
        self.cart_dr_range = cart_dr_range
        self.keypoint_range = keypoint_range
        self.ori_type = orientation_type
        self.trial_list = trial_list
        self.crop_radius = crop_radius
        self.crop_radius_shift = crop_radius_shift
        self.crop_radius_range = crop_radius_range
        self.crop_radius_chance = crop_radius_chance
        self._cr_min = crop_radius_range[0]
        self._cr_max = crop_radius_range[1]
        self._cr_rng = self._cr_max - self._cr_min
        self._ambiguous_radius = ambiguous_radius
        self.depth_factor = depth_factor
        self.proprio_dim = 3 + 4 + 1

        # super(RoboPenDataset, self).__init__(
        super(RLBenchDataset, self).__init__(
            dirname,
            template,
            verbose,
            trial_list=trial_list,
            TrialType=RPHighLevelTrial,
        )
        self._voxel_size = 0.001
        self._voxel_size_2 = 0.01
        self._local_problem_size = 0.1
        self.num_pts = num_pts
        self.DEBUG = True

        # configuration and data files
        self.cam_mapping_json_path = "./assets/robopen08_mapping.json"
        self.calibration_json_path = "./assets/robopen08_calibration.json"
        self.intrinsics_json_path = "./assets/robopen08_intrinsics.json"
        self.cam_intrinsics, self.cam_extrinsics = self.read_cam_config()

        self.debug_closest_pt = False
        if robot == "franka":
            temp = FrankaPanda()
            self._robot_ee_to_grasp_offset = temp.grasp_offset
            self._robot_max_grasp = temp.max_grasp
        elif robot == "stretch":
            # Offset from STRETCH_GRASP_FRAME to predicted grasp point
            self._robot_ee_to_grasp_offset = STRETCH_TO_GRASP.copy()
            self._robot_ee_to_grasp_offset[2, 3] -= 0.10
            self._robot_max_grasp = 0  # 0.13, empirically found
        else:
            raise ValueError("robot must be franka or stretch")
        self._robot = robot

        self.show_voxelized_input_and_reference = show_voxelized_input_and_reference
        self.show_input_and_reference = show_raw_input_and_reference
        self.show_cropped = show_cropped
        self.use_first_frame_as_input = first_frame_as_input
        self._visualize_interaction_estimates = visualize_interaction_estimates
        self._visualize_cropped_keyframes = visualize_cropped_keyframes

        # setup segmentation pipeline
        # self.segmentor = DeticPerception(
        #     vocabulary="custom",
        #     custom_vocabulary=",".join(REAL_WORLD_CATEGORIES),
        #     sem_gpu_id=0,
        # )

    def get_gripper_pose(self, trial, idx):
        """add grasp offset to ee pose and return gripper pose"""
        ee_pose = trial["ee_pose"][idx]
        pos = ee_pose[:3]
        x, y, z, w = ee_pose[3:]
        ee_pose = tra.quaternion_matrix([w, x, y, z])
        ee_pose[:3, 3] = pos
        ee_pose = ee_pose @ self._robot_ee_to_grasp_offset
        return ee_pose

    def get_gripper_axis(self, rot_mat):
        return rot_mat[:3, 2]

    def read_cam_config(self):
        """read camera intrinsics and extrinsics from json files"""
        with open(self.cam_mapping_json_path, "r") as f:
            cam_mapping = json.load(f)
        # with open(self.intrinsics_json_path, "r") as f:
        #     cam_intrinsics = json.load(f)
        with open(self.calibration_json_path, "r") as json_file:
            cam_extrinsic = json.load(json_file)
        cam_extrinsic_dict = {}
        cam_intrinsic_dict = {}
        for came in cam_extrinsic:
            cam_extrinsic_dict[
                cam_mapping["camera_mapping"][came["camera_serial_number"]]
            ] = came
            cam_intrinsic_dict[
                cam_mapping["camera_mapping"][came["camera_serial_number"]]
            ] = came["intrinsics"]
        # for camid, cami in cam_intrinsics.items():
        #     cam_intrinsic_dict[cam_mapping["camera_mapping"][camid]] = cami
        return cam_intrinsic_dict, cam_extrinsic_dict

    def process_images_from_view(
        self, trial: Trial, view_name: str, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """process rgb and depth image from a given camera into a structured PCD
        Args:
            trial:      Trial object
            view_name:  semantic name of the camera
                        expect images to be <view_name>_rgb, <view_name>_depth
            idx:        index of the image
        """
        rgb = trial.get_img(view_name + "_rgb", idx, rgb=True)
        depth = trial.get_img(
            view_name + "_depth", idx, depth=True, depth_factor=self.depth_factor
        )
        if self._robot == "stretch":
            xyz = trial[view_name + "_xyz"][idx]
        # rgb_img = rgb.copy()
        # depth_img = depth.copy()

        # get camera details
        if self._robot == "franka":
            camera_intrinsics = self.cam_intrinsics[view_name]
            if view_name != "wrist":
                camera_position = self.cam_extrinsics[view_name]["camera_base_pos"]
                camera_rot = self.cam_extrinsics[view_name]["camera_base_ori"]
            else:
                camera_position = self.cam_extrinsics[view_name]["camera_ee_pos"]
                camera_rot = self.cam_extrinsics[view_name]["camera_ee_ori"]
            padded_rot = np.concatenate((camera_rot, np.zeros((1, 3))))
            padded_trans = np.append(camera_position, 1.0)
            camera_matrix = np.concatenate(
                (padded_rot, padded_trans.reshape(-1, 1)), axis=1
            )

        if self.data_augmentation:
            depth = dropout_random_ellipses(depth, dropout_mean=10)
            # TODO: this would not work as expected esp when we combine multiple PCDs

        # convert rgb, depth to rgbd point-cloud
        if self._robot == "franka":
            camera = Camera(
                pos=camera_position,
                orn=camera_rot,
                height=camera_intrinsics["height"],
                width=camera_intrinsics["width"],
                fx=camera_intrinsics["fx"],
                fy=camera_intrinsics["fy"],
                px=camera_intrinsics["ppx"],
                py=camera_intrinsics["ppy"],
            )
            xyz = depth_to_xyz(depth, camera)
        if self.data_augmentation:
            xyz = add_additive_noise_to_xyz(
                xyz,
                valid_mask=depth > 0.1,
                gp_rescale_factor_range=[12, 20],
                gaussian_scale_range=[0.0, 0.001],
            )
        H, W, C = xyz.shape
        xyz = xyz.reshape(-1, C)

        if self._robot == "franka":
            # transform the resultant x,y,z to robot-frame
            # Now it is in world frame
            xyz = trimesh.transform_points(xyz, camera_matrix)
            # xyz = xyz.reshape(H, W, C)
            if view_name == "wrist":
                # transform from ee to the world frame
                # TODO: get ee-pose as a matrix
                ee_pose = trial["ee_pose"][idx]
                pos = ee_pose[:3]
                x, y, z, w = ee_pose[3:]
                ee_pose = tra.quaternion_matrix([w, x, y, z])
                ee_pose[:3, 3] = pos
                xyz = trimesh.transform_points(xyz, ee_pose)
        elif self._robot == "stretch":
            if "camera_pose" in trial.temporal_keys:
                camera_matrix = trial["camera_pose"][idx]
                xyz = trimesh.transform_points(xyz, camera_matrix)
            else:
                raise RuntimeError(
                    "Couldn't find camera information in your H5 file. The program will close now"
                )

        # downsample point-cloud by distance (heuristic)
        rgb = rgb.reshape(-1, C)
        depth = depth.reshape(-1)
        xyz = xyz.reshape(-1, C)
        mask = np.bitwise_and(depth < 1.5, depth > 0.3)
        rgb = rgb[mask]
        xyz = xyz[mask]
        return rgb, xyz

    def extract_manual_keyframes(self, user_keyframe_array):
        """returns indices of all user-tagged keyframes"""
        # return indices of all elements == 1
        user_keyframe_array = user_keyframe_array.squeeze()
        idx = np.arange(len(user_keyframe_array))
        idx = idx[user_keyframe_array == 1]
        return idx

    def choose_keypoint(
        self, keypoints: np.ndarray, keypoint_idx: int
    ) -> Tuple[int, int]:
        """return a randomly chosen keypoint from the list of keypoints;
        or return the one explicitly asked"""
        if self.keypoint_range is not None:
            chosen_idx = keypoint_idx % len(
                self.keypoint_range
            )  # each keypoint shows up TOTAL_KEYPOINTS / len(keypoint_range) times
            chosen_idx = self.keypoint_range[chosen_idx]
        else:
            chosen_idx = keypoint_idx % len(keypoints)
        return (
            keypoints[chosen_idx],
            chosen_idx,
        )  # actual keypoint index in the episode

    def show_cropped_keyframes(
        self, crop_xyz, crop_rgb, crop_ee_keyframe, crop_ref_ee_keyframe
    ):
        print(
            "Showing cropped PCD with original interaction-ee-position and current-ee-keyframe"
        )
        show_point_cloud_with_keypt_and_closest_pt(
            crop_xyz,
            crop_rgb,
            crop_ee_keyframe[:3, 3],
            crop_ee_keyframe[:3, :3],
            crop_ref_ee_keyframe[:3, 3],
        )
        print(
            "Showing cropped PCD with perturbed interaction-ee-position and current-ee-keyframe"
        )
        show_point_cloud_with_keypt_and_closest_pt(
            crop_xyz,
            crop_rgb,
            crop_ee_keyframe[:3, 3],
            crop_ee_keyframe[:3, :3],
            np.array([0, 0, 0]),
        )

    def show_interaction_point_and_keyframe(
        self,
        xyz2,
        rgb2,
        current_ee_keyframe,
        closest_pt_down_pcd,
        interaction_ee_keyframe,
    ):
        print(
            "Showing current ee keyframe as the coordinate-frame and the interaction-point in PCD as yellow sphere"
        )
        show_point_cloud_with_keypt_and_closest_pt(
            xyz2,
            rgb2,
            current_ee_keyframe[:3, 3],
            current_ee_keyframe[:3, :3],
            closest_pt_down_pcd,
        )
        print(
            "Showing current ee keyframe as the coordinate-frame and the interaction-ee-position as yellow sphere"
        )
        show_point_cloud_with_keypt_and_closest_pt(
            xyz2,
            rgb2,
            current_ee_keyframe[:3, 3],
            current_ee_keyframe[:3, :3],
            interaction_ee_keyframe[:3, 3],
        )

    def get_datum(self, trial, keypoint_idx, verbose=False):
        """Get a single training example given the index."""
        debug = False

        cmds = trial["task_name"][()].decode("utf-8").split(",")
        cmd = cmds[0]
        if verbose:
            print(f"{cmd=}")
        self.task_name = cmd
        self.h5_filename = trial.h5_filename
        if self.annotations is not None:
            cmd_opts = self.annotations[cmd]
            cmd = cmd_opts[np.random.randint(len(cmd_opts))]
        if self.per_action_cmd:
            # hard-coded for pick_up_bottle right now
            cmd = [
                "approach-pose-action bottle",
                "grasp-action bottle",
                "lift-action bottle",
            ]
        # TODO: remove this and read from a yaml file instead
        all_cmd = [
            "approach-pose-action bottle",
            "grasp-action bottle",
            "lift-action bottle",
        ]

        if self.skill_to_action is not None:
            breakpoint()
            all_cmd = self.skill_to_action[cmd]

        if self.skill_to_action is not None and self.per_action_cmd:
            """return different language per waypoint"""
            cmd = all_cmd

        if verbose:
            print(f"{cmd=}")

        keypoints = self.extract_manual_keyframes(
            trial["user_keyframe"][()]
        )  # list of index of keypts
        current_keypoint_idx, keypoint_relative_idx = self.choose_keypoint(
            keypoints, keypoint_idx
        )
        if self.per_action_cmd:
            cmd = cmd[keypoint_relative_idx]
        # create time-step
        if self.time_as_one_hot:
            # create time as a one-hot vector
            time_step = (
                torch.nn.functional.one_hot(
                    torch.LongTensor([keypoint_relative_idx]), self.max_keypoints
                )
                .numpy()
                .squeeze()
            )
        else:
            time_step = np.array(
                [(keypoint_relative_idx / (self.max_keypoints - 1) - 0.5) * 2]
            )

        if self.multi_step:
            num_keyframes = len(keypoints)
        else:
            num_keyframes = 1
        all_time_step = np.zeros((num_keyframes, self.max_keypoints))
        for idx in range(num_keyframes):
            if self.time_as_one_hot:
                all_time_step[idx] = (
                    torch.nn.functional.one_hot(
                        torch.LongTensor([idx]), self.max_keypoints
                    )
                    .numpy()
                    .squeeze()
                )
            else:
                all_time_step[idx] = np.array(
                    [(idx / (self.max_keypoints - 1) - 0.5) * 2]
                )
        # this index is of the actual episode step this keypoint belongs to; i.e. trial/current_keypoint_idx/<ee-pose, images, etc>
        if verbose:
            print(f"Key-point index chosen: abs={current_keypoint_idx}")

        gripper_width_array = trial["gripper_state"][()]
        if len(gripper_width_array.shape) == 1:
            num_samples = gripper_width_array.shape[0]
            gripper_width_array = gripper_width_array.reshape(num_samples, 1)
        gripper_state = (gripper_width_array <= self._robot_max_grasp).astype(int)
        interaction_pt_idx = -1
        for i, other_keypoint in enumerate(keypoints):
            interaction_pt_idx = other_keypoint
            if i == 0:
                continue
            if gripper_state[other_keypoint] != gripper_state[i - 1]:
                break

        if verbose:
            print(
                f"reference_pt: {interaction_pt_idx}, min_gripper: {self._robot_max_grasp}, gripper-state-array: {gripper_state}"
            )

        # choose an input frame-idx, in our case this is the 1st frame
        # associated with current keypoint
        # input_keyframes = self.extract_manual_keyframes(trial["input_keyframe"][()])

        # this array has more values when we are combining a trail of views leading up to the interaction
        num_input_frames = 1
        input_keyframes = []
        for i in range(num_input_frames):
            input_keyframes.append(keypoints[0] - i - 1)
        if self.use_first_frame_as_input:
            raise RuntimeError(
                "use_first_frame_as_input was used but it doesn't do anything right now"
            )
        #     image_index = keypoint_idx % 1
        #     # use one of the 1st two frames
        #     # TODO replace this with bursts of images around each keyframe which we can sample from
        #     # helps with overindexing on gripper/other unmoving objects
        # input_idx = np.concatenate([np.zeros(1), keypoints])[chosen_idx].astype(int)

        # the following should also be more consistent with stretch setup
        # TODO: verify data-loader works and remove the following
        # if chosen_idx == 2:
        #     input_idx = k_idx - 1
        # else:
        input_idx = input_keyframes[-1]  # query from the last frame
        if verbose:
            print(f"Index from where to query input state: {input_idx}")

        # get EE keyframe
        current_ee_keyframe = self.get_gripper_pose(trial, int(current_keypoint_idx))
        interaction_ee_keyframe = self.get_gripper_pose(trial, int(interaction_pt_idx))
        all_ee_keyframes = []
        if self.multi_step:
            target_gripper_state = np.zeros(len(keypoints))
            # Add all keypoints to this list
            for j, keypoint in enumerate(keypoints):
                all_ee_keyframes.append(self.get_gripper_pose(trial, int(keypoint)))
                target_gripper_state[j] = gripper_state[keypoint]
        else:
            # Pull out gripper state from the sim data
            for j, keypoint in enumerate(keypoints):
                all_ee_keyframes.append(self.get_gripper_pose(trial, int(keypoint)))
            target_gripper_state = gripper_state[current_keypoint_idx]

        # get point-cloud in base-frame from the cameras
        rgbs, xyzs = [], []
        for view in ["head"]:  # TODO: make keys consistent with stretch H5 schema
            for image_index in input_keyframes:
                v_rgb, v_xyz = self.process_images_from_view(
                    trial,
                    view,
                    image_index if image_index is not None else input_idx,
                )
                rgbs.append(v_rgb)
                xyzs.append(v_xyz)

        drop_frames = False  # TODO: get this from cfg
        if drop_frames:
            # randomly dropout 1/3rd of the point-clouds
            # TODO: update this to dropout each frame with 0.33 probability
            idx_dropout = np.random.choice(
                [False, True], size=len(rgbs), p=[0.33, 0.67]
            )
            rgbs = [rgbs[i] for i in idx_dropout]
            xyzs = [xyzs[i] for i in idx_dropout]
        rgb = np.concatenate(rgbs, axis=0)
        xyz = np.concatenate(xyzs, axis=0)
        x_mask = xyz[:, 0] < 0.9
        rgb = rgb[x_mask]
        xyz = xyz[x_mask]
        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)

        # using ee-keyframe at index interaction_pt_idx now compute point in PCD intersecting with action axis of the gripper
        gripper_pose = self.get_gripper_pose(trial, interaction_pt_idx)
        action_axis = self.get_gripper_axis(gripper_pose)
        gripper_position = gripper_pose[:3, 3]
        index, interaction_point = find_closest_point_to_line(
            xyz, gripper_position, action_axis
        )
        interaction_ee_keyframe[:3, 3] = interaction_point

        # voxelize at a granular voxel-size then choose X points
        xyz, rgb = self.remove_duplicate_points(xyz, rgb)
        xyz, rgb = self.dr_crop_radius(xyz, rgb, interaction_ee_keyframe)
        orig_xyz, orig_rgb = xyz, rgb

        # Get the point clouds and shuffle them around a bit
        xyz, rgb, center = self.shuffle_and_downsample_point_cloud(xyz, rgb)

        # mean-center the keyframes wrt classifier-input pcd
        orig_xyz -= center[None].repeat(orig_xyz.shape[0], axis=0)
        current_ee_keyframe[:3, 3] -= center
        interaction_ee_keyframe[:3, 3] -= center
        for keyframe in all_ee_keyframes:
            keyframe[:3, 3] -= center

        (
            orig_xyz,
            xyz,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        ) = self.dr_rotation_translation(
            orig_xyz,
            xyz,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        )

        (
            xyz2,
            rgb2,
            target_idx_down_pcd,
            closest_pt_down_pcd,
            target_idx_og_pcd,
            closest_pt_og_pcd,
        ) = self.voxelize_and_get_interaction_point(xyz, rgb, interaction_ee_keyframe)
        if xyz2 is None:
            print("Couldn't find an interaction point")
            return {"data_ok_status": False}

        # Get the local version of the problem
        (crop_location, crop_xyz, crop_rgb, data_status) = self.get_local_problem(
            orig_xyz, orig_rgb, closest_pt_down_pcd
        )
        if verbose:
            print(f"Size of cropped xyz: {crop_xyz.shape}")
        # Get data for the regression training
        # This needs to happen before centering i guess
        (
            crop_ref_ee_keyframe,
            crop_ee_keyframe,
            crop_keyframes,
        ) = self.get_local_commands(
            crop_location,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        )

        positions, orientations, angles = self.get_commands(
            crop_ee_keyframe, crop_keyframes
        )
        all_positions, all_orientation, all_angles = self.get_commands(
            crop_ee_keyframe, crop_keyframes, return_all=True
        )
        self._assert_positions_match_ee_keyframes(crop_ee_keyframe, positions)

        # create proprio vector
        if self.multi_step:
            all_proprio = np.zeros((len(keypoints), self.proprio_dim))
            for idx, key_idx in enumerate(keypoints):
                # all_proprio is (past_pos, past_quat, past_g, current_g, time)
                if idx == 0:
                    past_pos = 2 * np.ones(3)
                    past_quat = 2 * np.ones(4)
                    past_g = np.array([-1])
                else:
                    past_pos = all_positions[idx - 1]
                    past_quat = all_angles[idx - 1]
                    past_g = gripper_state[key_idx - 1]
                if verbose and debug:
                    print("Showing past-pos and quat")
                    show_point_cloud(
                        crop_xyz,
                        crop_rgb,
                        orig=past_pos.reshape(3, 1),
                        R=tra.quaternion_matrix(past_quat)[:3, :3],
                    )
                all_proprio[idx] = np.concatenate((past_pos, past_quat, past_g))

        if not self.autoregressive:
            proprio = np.concatenate(
                (gripper_state[input_idx], gripper_width_array[input_idx], time_step)
            )
        else:
            # proprio is (past_pos, past_quat, past_g, current_g, time)
            if keypoint_relative_idx == 0:
                past_pos = 2 * np.ones(3)
                past_quat = 2 * np.ones(4)
                past_g = np.array([-1])
            else:
                past_pos = all_positions[keypoint_relative_idx - 1]
                past_quat = all_angles[keypoint_relative_idx - 1]
                past_g = gripper_state[keypoint_relative_idx - 1]
            if verbose and debug:
                print("Showing past-pos and quat")
                show_point_cloud(
                    crop_xyz,
                    crop_rgb,
                    orig=past_pos.reshape(3, 1),
                    R=tra.quaternion_matrix(past_quat)[:3, :3],
                )
            proprio = np.concatenate((past_pos, past_quat, past_g))

        if verbose:
            print(f"Proprio: {proprio}")

        if self._visualize_interaction_estimates:
            self.show_interaction_point_and_keyframe(
                xyz2,
                rgb2,
                current_ee_keyframe,
                closest_pt_down_pcd,
                interaction_ee_keyframe,
            )
            print(
                "Showing current ee keyframe as the coordinate-frame and the interaction-point in PCD as yellow sphere"
            )
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                current_ee_keyframe[:3, 3],
                current_ee_keyframe[:3, :3],
                closest_pt_down_pcd,
            )
            print(
                "Showing current ee keyframe as the coordinate-frame and the interaction-ee-position as yellow sphere"
            )
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                current_ee_keyframe[:3, 3],
                current_ee_keyframe[:3, :3],
                interaction_ee_keyframe[:3, 3],
            )

        if self._visualize_cropped_keyframes:
            self.show_cropped_keyframes(
                crop_xyz, crop_rgb, crop_ee_keyframe, crop_ref_ee_keyframe
            )

        datum = {
            "trial_name": trial.name,
            "data_ok_status": data_status,
            "num_keypoints": len(keypoints),
            # ----------
            "ee_keyframe_pos": torch.FloatTensor(current_ee_keyframe[:3, 3]),
            "ee_keyframe_ori": torch.FloatTensor(current_ee_keyframe[:3, :3]),
            "proprio": torch.FloatTensor(proprio),
            "all_proprio": torch.FloatTensor(all_proprio),
            "time_step": torch.FloatTensor(time_step),
            "all_time_step": torch.FloatTensor(all_time_step),
            "target_gripper_state": torch.FloatTensor(target_gripper_state),
            "xyz": torch.FloatTensor(xyz),
            "rgb": torch.FloatTensor(rgb),
            "cmd": cmd,
            "all_cmd": all_cmd,
            "keypoint_idx": keypoint_relative_idx,
            # engineered features ----------------
            "closest_pos": torch.FloatTensor(closest_pt_og_pcd),
            "closest_pos_idx": torch.LongTensor([target_idx_og_pcd]),
            "closest_voxel": torch.FloatTensor(closest_pt_down_pcd),
            "closest_voxel_idx": torch.LongTensor([target_idx_down_pcd]),
            "xyz_downsampled": torch.FloatTensor(xyz2),
            "rgb_downsampled": torch.FloatTensor(rgb2),
            # used in pt_query.py; make sure this is being used with xyz_downsampled
            # TODO rename xyz_mask --> xyz_downsampled_mask to remove confusion
            "xyz_mask": torch.LongTensor(
                self.mask_voxels(xyz2, target_idx_down_pcd)
            ),  # @Priyam: I have no idea what this is
            # Crop inputs -----------------
            "rgb_crop": torch.FloatTensor(crop_rgb),
            "xyz_crop": torch.FloatTensor(crop_xyz),
            "crop_ref_ee_keyframe_pos": torch.FloatTensor(crop_ref_ee_keyframe[:3, 3]),
            "crop_ref_ee_keyframe_ori": torch.FloatTensor(crop_ref_ee_keyframe[:3, :3]),
            "perturbed_crop_location": torch.FloatTensor(crop_location),
            # Crop goals ------------------
            # Goals for regression go here
            "target_ee_keyframe_pos_crop": torch.FloatTensor(positions),
            "target_ee_keyframe_ori_crop": torch.FloatTensor(orientations),
            "target_ee_angles": torch.FloatTensor(angles),
        }
        return datum


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default="/home/priparashar/Development/icra/home_robot/data/robopen/mst/",
)
@click.option("--split", help="json file with train-test-val split")
@click.option(
    "--waypoint-language", help="yaml for skill-to-action lang breakdown", default=""
)
@click.option("-ki", "--k-index", default=[0], multiple=True)
@click.option("-r", "--robot", default="stretch")
def debug_get_datum(data_dir, k_index, split, robot, waypoint_language):
    if split:
        with open(split, "r") as f:
            train_test_split = yaml.safe_load(f)
    loader = RobotDataset(
        data_dir,
        template="*.h5",
        num_pts=8000,
        data_augmentation=True,
        crop_radius=True,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        # first_keypoint_only=True,
        # keypoint_range=[0],
        trial_list=train_test_split["train"] if split else [],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=True,
        show_cropped=True,
        verbose=False,
        multi_step=True,
        visualize_interaction_estimates=True,
        visualize_cropped_keyframes=True,
        robot=robot,
        autoregressive=True,
        time_as_one_hot=True,
        per_action_cmd=True,
        skill_to_action_file=None if waypoint_language == "" else waypoint_language,
    )
    for trial in loader.trials:
        print(f"Trial name: {trial.name}")
        for k_i in k_index:
            data = loader.get_datum(trial, k_i, verbose=True)


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default="/home/priparashar/robopen_h5s/larp/9tasks_woutclutter",
)
@click.option("--split", help="json file with train-test-val split")
@click.option("--template", default="*.h5")
@click.option("--robot", default="stretch")
def show_all_keypoints(data_dir, split, template, robot):
    """function which visualizes keypoints overlaid on initial frame, then
    visualizes the input frame for each keypoint with labeled interaction
    point overlaid"""
    if split:
        with open(split, "r") as f:
            train_test_split = yaml.safe_load(f)
    loader = RobotDataset(
        data_dir,
        template=template,
        num_pts=8000,
        data_augmentation=True,
        crop_radius=True,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        # first_keypoint_only=True,
        # keypoint_range=[0],
        trial_list=train_test_split["train"] if split else [],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=True,
        show_cropped=True,
        verbose=False,
        multi_step=False,
        visualize_interaction_estimates=True,
        visualize_cropped_keyframes=True,
        robot=robot,
        autoregressive=True,
        time_as_one_hot=True,
    )
    skip_names = ["30_11_2022_15_22_40"]
    for trial in loader.trials:
        print(f"Trial: {trial.name}")
        print(f"Task name: {trial.h5_filename}")
        if trial.name in skip_names:
            print("skipping as known bad trajectory")
        else:
            num_keypt = trial.num_keypoints
            for i in range(num_keypt):
                print("Keypoint requested: ", i)
                data = loader.get_datum(trial, i, verbose=True)
            # data = loader.get_datum(trial, 1, verbose=False)


if __name__ == "__main__":
    debug_get_datum()
    pass
