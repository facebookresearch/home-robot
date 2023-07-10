# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Tuple, Type

import h5py
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as T
import trimesh.transformations as tra
from PIL import Image
from slap_manipulation.utils.data_visualizers import (
    show_point_cloud_with_keypt_and_closest_pt,
)

from home_robot.utils.data_tools.loader import DatasetBase, Trial
from home_robot.utils.point_cloud import (
    add_additive_noise_to_xyz,
    dropout_random_ellipses,
    numpy_to_pcd,
    show_point_cloud,
)


class RLBHighLevelTrial(Trial):
    """handle a domain-randomized trial"""

    def __init__(
        self,
        name: str,
        h5_filename: str,
        dataset: Type[DatasetBase],
        group: h5py.Group,
    ):
        """
        Use group for initialization
        """
        super().__init__(name, h5_filename, dataset, group)
        self.factor = 1
        self.dr_factor = 10
        num_samples = len(self["keypoints"]) if not dataset.multi_step else 1
        self.length = (
            self.factor
            * num_samples
            * (self.dr_factor if dataset.data_augmentation else 1)
        )


class RLBenchDataset(DatasetBase):
    """train on a dataset from RLBench"""

    def __init__(
        self,
        dirname,
        template="*.h5",
        trial_list: list = None,
        verbose: bool = False,
        num_pts: int = 10000,
        data_augmentation: bool = True,
        random_idx: bool = False,
        random_cmd: bool = True,
        first_keypoint_only: bool = False,
        ori_dr_range: float = np.pi / 4,
        cart_dr_range: float = 1.0,
        debug_closest_pt: bool = False,
        crop_radius: bool = True,
        crop_radius_chance: float = 0.75,
        crop_radius_shift: float = 0.05,
        crop_radius_range: List[float] = [1.0, 2.0],
        ambiguous_radius: float = 0.03,
        orientation_type: str = "rpy",
        multi_step: bool = False,
        color_jitter: bool = True,
        query_radius: float = 0.1,
        num_crop_tries: int = 10,
        min_num_pts: int = 50,
        *args,
        **kwargs,
    ):
        self._num_crop_tries = num_crop_tries
        self._min_num_pts = min_num_pts
        self.multi_step = multi_step
        self.query_radius = query_radius
        self.ori_type = orientation_type
        self.random_idx = random_idx
        self.random_cmd = random_cmd
        self.first_keypoint_only = first_keypoint_only
        self.keypoint_to_use = None
        self.num_pts = num_pts
        self.data_augmentation = data_augmentation
        if color_jitter:
            self.color_jitter = T.ColorJitter(
                brightness=0.25, hue=0.05, saturation=0.1, contrast=0.1
            )
        else:
            self.color_jitter = None
        self.ori_dr_range = ori_dr_range
        self.cart_dr_range = cart_dr_range
        self.crop_radius = crop_radius
        self.crop_radius_shift = crop_radius_shift
        self.crop_radius_range = crop_radius_range
        self.crop_radius_chance = crop_radius_chance
        self._cr_min = crop_radius_range[0]
        self._cr_max = crop_radius_range[1]
        self._cr_rng = self._cr_max - self._cr_min
        self._ambiguous_radius = ambiguous_radius
        super(RLBenchDataset, self).__init__(
            dirname, template, verbose, trial_list, TrialType=RLBHighLevelTrial
        )
        self._voxel_size = 0.001
        self._voxel_size_2 = 0.01
        self._local_problem_size = 0.1

        self.debug_closest_pt = debug_closest_pt
        self.task_name = ""
        self.h5_filename = ""

    def normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """make sure rgb values are b/w 0 to 1"""
        rgb = rgb / 255.0
        return rgb

    def get_gripper_pose(self, trial: RLBHighLevelTrial, idx: int) -> np.ndarray:
        """get the 6D gripper pose at a particular time-step"""
        pos = trial["ee_xyz"][idx]
        x, y, z, w = trial["ee_rot"][idx]
        ee_pose = tra.quaternion_matrix([w, x, y, z])
        ee_pose[:3, 3] = pos
        return ee_pose

    def mask_voxels(self, voxels: np.ndarray, query_pt_idx: int) -> np.ndarray:
        """return a mask telling us which voxels are not ambiguous,
        i.e. distance from :query_pt: is > self._ambiguous_radius"""
        query_pt = voxels[query_pt_idx]
        query = query_pt[None].repeat(voxels.shape[0], axis=0)
        dists = np.linalg.norm(voxels - query, axis=-1)
        mask = dists > self._ambiguous_radius
        mask[query_pt_idx] = True
        return mask

    def process_images_from_view(
        self, trial: RLBHighLevelTrial, view_name: str, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Processes images from a particular view given its name and index of
        image to read"""
        rgb = trial[view_name + "_rgb"][idx]
        if self.data_augmentation and self.color_jitter is not None:
            pil_img = Image.fromarray(rgb)
            pil_img = self.color_jitter(pil_img)
            rgb = np.array(pil_img)

        depth = trial[view_name + "_depth"][idx]
        xyz = trial[view_name + "_xyz"][idx]

        if self.data_augmentation:
            # This really doesnt do anything
            # depth = add_multiplicative_noise(depth, 10000, 0.0001)
            depth = dropout_random_ellipses(depth, dropout_mean=10)

        # Process information here
        mask = np.bitwise_and(depth > 0.1, depth < 3.0)
        mask2 = np.bitwise_and(xyz[:, :, 0] > -0.5, xyz[:, :, 2] > 0.5)
        mask = np.bitwise_and(mask, mask2)

        if self.data_augmentation:
            xyz = add_additive_noise_to_xyz(
                xyz,
                valid_mask=mask,
                gp_rescale_factor_range=[12, 20],
                gaussian_scale_range=[0.0, 0.001],
                # gaussian_scale_range=[0.0, 0.0001],
            )

        mask = mask.reshape(-1)
        rgb_pts = rgb.reshape(-1, 3)[mask]
        xyz_pts = xyz.reshape(-1, 3)[mask]

        return rgb_pts, xyz_pts

    def crop_around_voxel(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        feat: np.ndarray,
        voxel: np.ndarray,
        crop_size: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop a point cloud around given :voxel: (3D point) with radius
        :crop_size:
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feats: additional features per point (semantic features from Detic in
        this implementation)
        """
        mask = np.linalg.norm(xyz - voxel, axis=1) < crop_size
        return xyz[mask], rgb[mask], feat[mask]

    def mean_center_shuffle_and_downsample_point_cloud(
        self, xyz: np.ndarray, rgb: np.ndarray, feat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """mean center, shuffle and then sample :self.num_pts: from given
        point cloud described by
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feats: additional features per point (semantic features from Detic in
        this implementation)
        """
        # Downsample pt clouds
        downsample = np.arange(rgb.shape[0])
        np.random.shuffle(downsample)
        if self.num_pts != -1:
            downsample = downsample[: self.num_pts]
        rgb = rgb[downsample]
        xyz = xyz[downsample]
        feat = feat[downsample]

        # mean center xyz
        center = np.mean(xyz, axis=0)
        # center = np.zeros(3)
        # center[-1] = 0
        xyz = xyz - center[None].repeat(xyz.shape[0], axis=0)
        return xyz, rgb, feat, center

    def remove_duplicate_points(
        self, xyz: np.ndarray, rgb: np.ndarray, feat: np.ndarray, feat_dim=1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """remove duplicate points from point cloud by voxelizing with resolution
        :self._voxel_size:
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feats: additional features per point (semantic features from Detic in
        this implementation)
        """
        debug_views = False
        if debug_views:
            print("xyz", xyz.shape)
            print("rgb", rgb.shape)
            show_point_cloud(xyz, rgb)

        xyz, rgb, feat = (
            xyz.reshape(-1, 3),
            rgb.reshape(-1, 3),
            feat.reshape(-1, feat_dim),
        )
        # voxelize at a granular voxel-size rather than random downsample
        pcd = numpy_to_pcd(xyz, rgb)
        pcd_downsampled = pcd.voxel_down_sample(self._voxel_size)
        rgb = np.asarray(pcd_downsampled.colors)
        xyz = np.asarray(pcd_downsampled.points)

        debug_voxelization = False
        if debug_voxelization:
            show_point_cloud(xyz, rgb)

        return xyz, rgb, feat

    def dr_crop_radius(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        feat: np.ndarray,
        ref_ee_keyframe: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Radius crop around the ref_ee_keyframe with some probability
        :self.crop_radius_chance: with radius :self.crop_radius: as data-augmentation
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feat: additional features per point (semantic features from Detic in
        this implementation)
        """
        if self.data_augmentation and self.crop_radius:
            # crop out random points outside a certain distance from the gripper
            # this is to encourage it to learn only local features and skills
            if np.random.random() < self.crop_radius_chance:
                # Now we do the cropping
                orig = ref_ee_keyframe[:3, 3][None].copy()
                crop_shift = ((np.random.random(3) * 2) - 1) * self.crop_radius_shift
                orig += crop_shift
                # Now here we apply some other stuff
                orig = np.repeat(orig, xyz.shape[0], axis=0)
                crop_dist = np.linalg.norm(xyz - orig, axis=-1)
                radius = (np.random.random() * self._cr_rng) + self._cr_min
                crop_idx = crop_dist < radius
                rgb = rgb[crop_idx, :]
                xyz = xyz[crop_idx, :]
                feat = feat[crop_idx, :]
        return xyz, rgb, feat

    def voxelize_and_get_interaction_point(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        feat: np.ndarray,
        interaction_ee_keyframe: np.ndarray,
    ):
        """uniformly voxelizes the input point-cloud and returns the closest-point
        in the point-cloud to the task's interaction ee-keyframe
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feat: additional features per point (semantic features from Detic in
        this implementation)
        """
        # downsample another time to get sampled version
        pcd_downsampled = numpy_to_pcd(xyz, rgb)
        pcd_downsampled2 = pcd_downsampled.voxel_down_sample(self._voxel_size_2)
        xyz2 = np.asarray(pcd_downsampled2.points)
        rgb2 = np.asarray(pcd_downsampled2.colors)
        feat2 = feat

        # for the voxelized pcd
        if xyz2.shape[0] < 10:
            return (None, None, None, None, None, None)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_downsampled2)
        # Find closest points based on ref_ee_keyframe
        # This is used to supervise the location when we're detecting where the action
        # could have happened
        [_, target_idx_1, _] = pcd_tree.search_knn_vector_3d(
            interaction_ee_keyframe[:3, 3], 1
        )
        target_idx_down_pcd = np.asarray(target_idx_1)[0]
        closest_pt_down_pcd = xyz2[target_idx_down_pcd]

        # this is for exact point
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_downsampled)
        [_, target_idx_2, _] = pcd_tree.search_knn_vector_3d(
            # ee_keyframe[:3, 3], 1
            interaction_ee_keyframe[:3, 3],
            1,
        )
        target_idx_og_pcd = np.asarray(target_idx_2)[0]
        closest_pt_og_pcd = xyz[target_idx_og_pcd]

        if self.debug_closest_pt:
            print("Closest point in downsampled pcd")
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                interaction_ee_keyframe[:3, 3],
                interaction_ee_keyframe[:3, :3],
                xyz2[target_idx_down_pcd].reshape(3, 1),
            )
            print("Closest point in original pcd")
            show_point_cloud_with_keypt_and_closest_pt(
                xyz,
                rgb,
                interaction_ee_keyframe[:3, 3],
                interaction_ee_keyframe[:3, :3],
                xyz[target_idx_og_pcd].reshape(3, 1),
            )
        return (
            xyz2,
            rgb2,
            feat2,
            target_idx_down_pcd,
            closest_pt_down_pcd,
            target_idx_og_pcd,
            closest_pt_og_pcd,
        )

    def get_commands(
        self,
        crop_ee_keyframe: np.ndarray,
        keyframes: List[np.ndarray],
        return_all=False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Given the ee-keyframe in cropped frame-of-reference returns pos, rot_mat and quat ori
        associated with it
        Args:
            crop_ee_keyframe: np.ndarray of the keyframe as a 4D matrix
            keyframes: list of 4d homogeneous matrix
            return_all: overrides OFF state of `autoregressive` setting,
                expects keyframes to have all the keyframes as input
        Return:
            positions: List of positions of keyframes/crop_ee_keyframe
            orientations: List of rotation_matrix associated with keyframes/crop_ee_keyframe
            angles: quaternion/rpy associated with keyframes/crop_ee_keyframe
        """
        if self.multi_step or return_all:
            num_frames = len(keyframes)
            assert num_frames > 0
            positions = np.zeros((num_frames, 3))
            orientations = np.zeros((num_frames, 3, 3))
            # Set things up with the right shape
            if self.ori_type == "rpy":
                angles = np.zeros((num_frames, 3))
            elif self.ori_type == "quaternion":
                angles = np.zeros((num_frames, 4))
            else:
                raise RuntimeError(
                    "unsupported orientation type: " + str(self.ori_type)
                )
            # Loop over the whole list of keyframes
            # Create a set of trajectory data so we can train all three waypoints at once
            for j, keyframe in enumerate(keyframes):
                orientations[j, :, :] = keyframe[:3, :3]
                positions[j, :] = keyframe[:3, 3]
                if self.ori_type == "rpy":
                    angles[j, :] = tra.euler_from_matrix(keyframe[:3, :3])
                elif self.ori_type == "quaternion":
                    angles[j, :] = tra.quaternion_from_matrix(keyframe[:3, :3])
        else:
            # Just one
            positions = [crop_ee_keyframe[:3, 3]]
            orientations = [crop_ee_keyframe[:3, :3]]
            if self.ori_type == "rpy":
                angles = [tra.euler_from_matrix(crop_ee_keyframe[:3, :3])]
            elif self.ori_type == "quaternion":
                angles = [tra.quaternion_from_matrix(crop_ee_keyframe[:3, :3])]
        return positions, orientations, angles

    def get_local_problem(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        feat: np.ndarray,
        interaction_pt: np.ndarray,
        query_radius: float = None,
        num_find_crop_tries: int = None,
        min_num_points: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Crop given PCD around :interaction_pt: as input to action
        prediction problem, after perturbing :interaction_pt: by some small amt
        Cropping is retried :num_find_crop_tries: times, until :min_num_points:
        are found around (:interaction_pt: + noise)

        Returns the location around which the crop was made,
        cropped xyz, rgb, feat and a boolean indicating whether the crop was
        successful or not
        :xyz: 3D positions of points in point-cloud
        :rgb: rgb values of points in point-cloud
        :feat: additional features per point (semantic features from Detic in
        this implementation)
        """
        if query_radius is None:
            query_radius = self.query_radius
        if num_find_crop_tries is None:
            num_find_crop_tries = self._num_crop_tries
        if min_num_points is None:
            min_num_points = self._min_num_pts
        orig_crop_location = interaction_pt
        if self.data_augmentation:
            # Check to see if enough points are within the crop radius
            for _ in range(num_find_crop_tries):
                crop_location = orig_crop_location
                # Crop randomly within a few centimeters
                crop_location = orig_crop_location + (
                    (np.random.random(3) * 0.05) - 0.025
                )
                # Make sure at least min_num_points are within this radius
                # get number of points in this area
                dists = np.linalg.norm(
                    xyz - crop_location[None].repeat(xyz.shape[0], axis=0),
                    axis=-1,
                )
                # Make sure this is near some geometry
                if np.sum(dists < query_radius) > min_num_points:
                    break
                else:
                    crop_location = orig_crop_location
        else:
            crop_location = orig_crop_location

        # crop from og pcd and mean-center it
        crop_xyz, crop_rgb, crop_feat = self.crop_around_voxel(
            xyz, rgb, feat, crop_location, self._local_problem_size
        )
        crop_xyz = crop_xyz - crop_location[None].repeat(crop_xyz.shape[0], axis=0)
        # show_point_cloud(crop_xyz, crop_rgb, orig=np.zeros(3))
        if crop_rgb.shape[0] > self.num_pts:
            # Downsample pt clouds
            downsample = np.arange(crop_rgb.shape[0])
            np.random.shuffle(downsample)
            if self.num_pts != -1:
                downsample = downsample[: self.num_pts]
            crop_rgb = crop_rgb[downsample]
            crop_xyz = crop_xyz[downsample]
            crop_feat = crop_feat[downsample]
        status = True
        if crop_xyz.shape[0] < 10:
            status = False
        return crop_location, crop_xyz, crop_rgb, crop_feat, status

    def get_local_commands(
        self,
        crop_location: np.ndarray,
        ee_keyframe: np.ndarray,
        ref_ee_keyframe: np.ndarray,
        keyframes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns target action  by transforming it wrt :crop_location: as well as
        transforms each action in :keyframes: to be wrt :crop_location:
        """
        # NOTE: copying the keyframes is EXTREMELY important
        crop_ee_keyframe = ee_keyframe.copy()
        crop_ee_keyframe[:3, 3] -= crop_location
        crop_ref_ee_keyframe = ref_ee_keyframe.copy()
        crop_ref_ee_keyframe[:3, 3] -= crop_location
        crop_keyframes = []
        for keyframe in keyframes:
            _keyframe = keyframe.copy()
            _keyframe[:3, 3] -= crop_location
            crop_keyframes.append(_keyframe)
        return crop_ref_ee_keyframe, crop_ee_keyframe, crop_keyframes

    def _assert_positions_match_ee_keyframes(
        self,
        crop_ee_keyframe: np.ndarray,
        positions: np.ndarray,
        tol: float = 1e-6,
    ):
        """sanity check to make sure our data pipeline is consistent with multi head vs
        single channel data"""
        for pos in positions:
            err = np.linalg.norm(crop_ee_keyframe[:3, 3] - pos)
            if err < tol:
                return True
        else:
            raise RuntimeError(
                "crop ee keyframe and keypoints do not line up; "
                "you have a logic error"
            )

    def dr_rotation_translation(
        self,
        orig_xyz: np.ndarray,
        xyz: np.ndarray,
        ee_keyframe: np.ndarray,
        ref_ee_keyframe: np.ndarray,
        keyframes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Translate in 3D and then rotate entire PCD+interaction_point+actions
        around z-axis"""
        # note: above transforms points wrt translation and rotation provided
        # the second argument is a homogeneous matrix
        if self.data_augmentation:
            # Now that it is mean centered, apply data augmentation
            # Start with rotation
            rotation_matrix = tra.euler_matrix(
                0, 0, self.ori_dr_range * ((np.random.rand() * 2) - 1)
            )
            orig_xyz = tra.transform_points(orig_xyz, rotation_matrix)
            xyz = tra.transform_points(xyz, rotation_matrix)
            ee_keyframe = rotation_matrix @ ee_keyframe
            ref_ee_keyframe = rotation_matrix @ ref_ee_keyframe

            # Now add a random shift
            shift = ((np.random.rand(3) * 2) - 1) * self.cart_dr_range

            # Add shift to everything... yeah this could be written better
            ref_ee_keyframe[:3, 3] += shift
            xyz += shift[None].repeat(xyz.shape[0], axis=0)
            orig_xyz += shift[None].repeat(orig_xyz.shape[0], axis=0)
            ee_keyframe[:3, 3] += shift

            # Cropped trajectories
            # Loop over keyframes
            new_keyframes = []
            for keyframe in keyframes:
                keyframe = keyframe.copy()
                keyframe = rotation_matrix @ keyframe
                keyframe[:3, 3] += shift
                new_keyframes.append(keyframe)
        else:
            new_keyframes = keyframes
        return (orig_xyz, xyz, ee_keyframe, ref_ee_keyframe, new_keyframes)

    def get_datum(self, trial: RLBHighLevelTrial, keypoint_idx: int) -> Dict[str, Any]:
        """Get a single training example given the index."""

        # Idx is going to determine keypoint, not actual index
        keypoints = trial["keypoints"][()]  # list of index of keypts
        if self.first_keypoint_only:
            keypoint_idx = 0
            if self.keypoint_to_use is not None:
                keypoint_idx = self.keypoint_to_use
        else:
            keypoint_idx = keypoint_idx % len(keypoints)
        next_keypoint = trial["next_keypoint"][()]
        keypoint = keypoints[keypoint_idx]
        proprio = trial["low_dim_state"][()]

        dist = float("Inf")

        # Go through the keypoints
        for i, other_keypoint in enumerate(keypoints):
            if i == 0:
                continue
            other_reference_pt = other_keypoint
            if proprio[other_keypoint][0] != proprio[other_keypoint - 1][0]:
                break
            # Check to see if gripper was closed
            # Also compute distance to keypoint
            dist = np.abs(keypoint - other_keypoint)
        else:
            other_reference_pt = keypoint

        # get list of all indices between previous and next keypoint
        # # wrt chosen next_keypoint
        all_idxs = np.arange(len(next_keypoint))[next_keypoint == keypoint]
        idx = np.min(all_idxs)

        # ----------------------------
        # create point clouds
        rgbs, xyzs = [], []
        for view in ["overhead", "left", "right", "front"]:
            v_rgb, v_xyz = self.process_images_from_view(trial, view, idx)
            rgbs.append(self.normalize_rgb(v_rgb))
            xyzs.append(v_xyz)
        rgb = np.concatenate(rgbs, axis=0)
        xyz = np.concatenate(xyzs, axis=0)
        # ----------------------------

        k_idx = next_keypoint[idx]
        if k_idx != keypoint:
            print("WARNING; mismatch:", k_idx, keypoint, "at", idx)

        # get EE keyframe
        ref_ee_keyframe = self.get_gripper_pose(trial, other_reference_pt)
        ee_keyframe = self.get_gripper_pose(trial, int(k_idx))
        keyframes = []
        if self.multi_step:
            target_gripper_state = np.zeros(len(keypoints))
            # Add all keypoints to this list
            for j, keypoint in enumerate(keypoints):
                keyframes.append(self.get_gripper_pose(trial, int(keypoint)))
                target_gripper_state[j] = proprio[keypoint][0]
        else:
            # Pull out gripper state from the sim data
            target_gripper_state = proprio[k_idx][0]

        # Reduce the size of the point cloud further
        xyz, rgb = self.remove_duplicate_points(xyz, rgb)
        xyz, rgb = self.dr_crop_radius(xyz, rgb, ref_ee_keyframe)
        orig_xyz, orig_rgb = xyz, rgb

        # Get the point clouds and shuffle them around a bit
        xyz, rgb, center = self.mean_center_shuffle_and_downsample_point_cloud(xyz, rgb)

        # adjust our keyframes
        orig_xyz -= center[None].repeat(orig_xyz.shape[0], axis=0)
        ee_keyframe[:3, 3] -= center
        ref_ee_keyframe[:3, 3] -= center
        for keyframe in keyframes:
            keyframe[:3, 3] -= center

        (
            orig_xyz,
            xyz,
            ee_keyframe,
            ref_ee_keyframe,
            keyframes,
        ) = self.dr_rotation_translation(
            orig_xyz, xyz, ee_keyframe, ref_ee_keyframe, keyframes
        )

        (
            xyz2,
            rgb2,
            target_idx_down_pcd,
            closest_pt_down_pcd,
            target_idx_og_pcd,
            closest_pt_og_pcd,
        ) = self.voxelize_and_get_interaction_point(xyz, rgb, ref_ee_keyframe)

        # Get the local version of the problem
        (crop_location, crop_xyz, crop_rgb) = self.get_local_problem(
            orig_xyz, orig_rgb, closest_pt_down_pcd
        )
        # Get data for the regression training
        # This needs to happen before centering i guess
        (
            crop_ref_ee_keyframe,
            crop_ee_keyframe,
            crop_keyframes,
        ) = self.get_local_commands(
            crop_location, ee_keyframe, ref_ee_keyframe, keyframes
        )

        # Get the commands we care about here
        positions, orientations, angles = self.get_commands(
            crop_ee_keyframe, crop_keyframes
        )

        # predict vector from each point in PCD to the contact/release point
        cmds = trial["descriptions"][()].decode("utf-8").split(",")
        if self.random_cmd:
            cmd = cmds[np.random.randint(len(cmds))]
        else:
            cmd = cmds[0]

        proprio = np.array([keypoint_idx / len(keypoints) - 1])
        proprio = np.concatenate((trial["low_dim_state"][()][0][:-1], proprio))

        datum = {
            "trial_name": trial.name,
            "ee_keyframe_pos": torch.FloatTensor(ee_keyframe[:3, 3]),
            "ee_keyframe_ori": torch.FloatTensor(ee_keyframe[:3, :3]),
            "xyz": torch.FloatTensor(xyz),
            "rgb": torch.FloatTensor(rgb),
            "cmd": cmd,
            "keypoint_idx": keypoint_idx,
            "num_keypoints": len(keypoints),
            "proprio": torch.FloatTensor(trial["low_dim_state"][()][0][:-1]),
            "closest_pos": torch.FloatTensor(closest_pt_og_pcd),
            "closest_pos_idx": torch.LongTensor([target_idx_og_pcd]),
            "closest_voxel": torch.FloatTensor(closest_pt_down_pcd),
            "closest_voxel_idx": torch.LongTensor([target_idx_down_pcd]),
            "xyz_downsampled": torch.FloatTensor(xyz2),
            "rgb_downsampled": torch.FloatTensor(rgb2),
            "target_gripper_state": torch.FloatTensor([target_gripper_state]),
            "xyz_mask": torch.LongTensor(self.mask_voxels(xyz2, target_idx_down_pcd)),
            # Crop info ------
            "rgb_crop": torch.FloatTensor(crop_rgb),
            "xyz_crop": torch.FloatTensor(crop_xyz),
            "crop_ref_ee_keyframe_pos": crop_ref_ee_keyframe[:3, 3],
            # Crop goals ------------------
            # Goals for regression go here
            "ee_keyframe_pos_crop": torch.FloatTensor(positions),
            "ee_keyframe_ori_crop": torch.FloatTensor(orientations),
            "target_ee_angles": torch.FloatTensor(angles),
        }
        return datum


def debug_get_datum():
    loader = RLBenchDataset(
        "./data/rlbench",
        data_augmentation=False,
        first_keypoint_only=True,
        debug_closest_pt=True,
    )
    num_keypts = len(loader.trials[0]["keypoints"][()])
    for trial in loader.trials:
        for keypt_idx in range(num_keypts):
            loader.keypoint_to_use = keypt_idx
            _ = loader.get_datum(trial, 5)


if __name__ == "__main__":
    debug_get_datum()
