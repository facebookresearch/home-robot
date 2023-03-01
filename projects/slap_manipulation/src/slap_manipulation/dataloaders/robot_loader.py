import json
import math
from typing import List, Optional, Sequence, TypeVar, Union

import click
import numpy as np
import open3d as o3d
import torch
import trimesh
import trimesh.transformations as tra
import yaml
from home_robot.utils.data_tools.camera import Camera
from home_robot.utils.data_tools.loader import Trial
from home_robot.utils.point_cloud import (
    add_additive_noise_to_xyz,
    depth_to_xyz,
    dropout_random_ellipses,
    numpy_to_pcd,
)

from slap_manipulation.dataloaders.annotations import load_annotations_dict
from slap_manipulation.dataloaders.rlbench_loader import RLBenchDataset
# TODO Replace with Stretch embodiment
from home_robot.motion.franka import FrankaPanda


def show_point_cloud_with_keypt_and_closest_pt(
    xyz, rgb, keypt_orig, keypt_rot, closest_pt
):
    if np.any(rgb) > 1:
        rgb = rgb / 255.0
    pcd = numpy_to_pcd(xyz, rgb)
    geoms = [pcd]
    if keypt_orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=keypt_orig
        )
        if keypt_rot is not None:
            coords = coords.rotate(keypt_rot)
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

    def __init__(self, name, h5_filename, dataset, group):
        """
        Use group for initialization
        """
        super().__init__(name, h5_filename, dataset, group)
        keypoint_array = self["user_keyframe"][()].squeeze()
        idx = np.arange(keypoint_array.shape[0])
        idx = idx[keypoint_array == 1]
        keypoint_len = len(idx)
        # extra samples for metrics - used to coer for randomness in ptnet ops?
        self.factor = 1
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
    """train on a dataset from RLBench"""

    def __init__(
        self,
        dirname,
        template="*.h5",
        predict: str = "",
        verbose=False,
        num_pts=10000,
        data_augmentation=True,
        random_idx=False,
        random_cmd=True,
        first_keypoint_only=False,
        keypoint_range: list = [0, 1, 2],
        show_voxelized_input_and_reference=False,
        show_input_and_reference=False,
        show_cropped=False,
        ori_dr_range=np.pi / 4,
        cart_dr_range=1.0,
        first_frame_only=False,
        trial_list: list = [],
        orientation_type="quaternion",
        multi_step=False,
        crop_radius=True,
        ambiguous_radius=0.03,
        crop_radius_chance=0.75,
        crop_radius_shift=0.1,
        crop_radius_range=[0.3, 1.0],
        visualize=False,
        visualize_reg_targets=False,
        yaml_file="./assets/language_variations/v0.yml",
        dr_factor=1,
        *args,
        **kwargs,
    ):
        if yaml_file is not None:
            self.annotations = load_annotations_dict(yaml_file)
        else:
            self.annotations = None
        self.random_idx = random_idx
        self.random_cmd = random_cmd
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
        self.predict = predict  # can be "contact" or "release"

        # configuration and data files
        self.cam_mapping_json_path = "./assets/robopen08_mapping.json"
        self.calibration_json_path = "./assets/robopen08_calibration.json"
        self.intrinsics_json_path = "./assets/robopen08_intrinsics.json"
        self.cam_intrinsics, self.cam_extrinsics = self.read_cam_config()
        self.task_var_path = "./assets/robopen/task_variations.yaml"
        self.task_variations = self.read_tasks(self.task_var_path)

        self.debug_closest_pt = False
        self.robot = FrankaPanda()

        self.show_voxelized_input_and_reference = show_voxelized_input_and_reference
        self.show_input_and_reference = show_input_and_reference
        self.show_cropped = show_cropped
        self.use_first_frame_as_input = first_frame_only
        self.visualize = visualize
        self.visualize_reg_targets = visualize_reg_targets

    def get_gripper_pose(self, trial, idx):
        ee_pose = trial["ee_pose"][idx]
        pos = ee_pose[:3]
        x, y, z, w = ee_pose[3:]
        ee_pose = tra.quaternion_matrix([w, x, y, z])
        ee_pose[:3, 3] = pos
        ee_pose = self.robot.apply_grasp_offset(ee_pose)
        return ee_pose

    def read_tasks(self, path):
        with open(path, "r") as f:
            task_variations = yaml.load(f, Loader=yaml.FullLoader)
        return task_variations

    def read_cam_config(self):
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

    def process_images_from_view(self, trial, view_name, idx):
        rgb = trial.get_img(view_name + "_rgb", idx, rgb=True)
        depth = trial.get_img(view_name + "_depth", idx,
                              depth=True, depth_factor=1000)

        # get camera details
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

        # convert rgb, depth to rgbd point-cloud
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

        # transform the resultant x,y,z to robot-frame
        H, W, C = xyz.shape
        xyz = xyz.reshape(-1, C)
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

        # downsample point-cloud by distance (heuristic)
        # TODO get mask from mdetr
        rgb = rgb.reshape(-1, C)
        depth = depth.reshape(-1)
        mask = np.bitwise_and(depth < 1.5, depth > 0.3)
        rgb = rgb[mask]
        xyz = xyz[mask]

        return rgb, xyz

    def extract_manual_keyframes(self, user_keyframe_array):
        """returns indices of all keyframes"""
        # return indices of all elements == 1
        user_keyframe_array = user_keyframe_array.squeeze()
        idx = np.arange(len(user_keyframe_array))
        idx = idx[user_keyframe_array == 1]
        return idx

    def get_datum(self, trial, keypoint_idx, verbose=False):
        """Get a single training example given the index."""

        cmds = trial["task_name"][()].decode("utf-8").split(",")
        if self.random_cmd:
            cmd = cmds[np.random.randint(len(cmds))]
        else:
            cmd = cmds[0]
        if verbose:
            print(f"{cmd=}")
        self.task_name = cmd
        self.h5_filename = trial.h5_filename
        if self.annotations is not None:
            cmd_opts = self.annotations[cmd]
            # print(cmd_opts)
            cmd = cmd_opts[np.random.randint(len(cmd_opts))]
            # print(cmd)
        # else:
        # raise RuntimeError('we are tryin gto use the annotation file')
        # breakpoint()

        keypoints = self.extract_manual_keyframes(
            trial["user_keyframe"][()]
        )  # list of index of keypts
        if self.first_keypoint_only:
            chosen_idx = 0  # choosing index of keypoints
            if self.keypoint_to_use is not None:
                chosen_idx = self.keypoint_to_use
        else:
            if self.keypoint_range is not None:
                chosen_idx = keypoint_idx % len(self.keypoint_range)
                chosen_idx = self.keypoint_range[chosen_idx]
            else:
                # skip the 1st keyframe; 'tis the input pair to 1st actual keyframe
                chosen_idx = keypoint_idx % len(keypoints)
        k_idx = keypoints[chosen_idx]  # actual keypoint index in the episode
        if verbose:
            print(
                f"Key-point index chosen: abs={k_idx}, relative={chosen_idx}")

        # create proprio features
        proprio = trial["gripper_state"][()]
        min_gripper = 0.95 * 0.08  # 95% of max_width=8cm
        gripper_state = (proprio <= min_gripper).astype(int)

        reference_pt = -1
        if "place" in cmd or "put" in cmd or "add" in cmd:
            # TODO move to configs; or get better data
            reference_pt = keypoints[1]
        else:
            for i, other_keypoint in enumerate(keypoints):
                reference_pt = other_keypoint
                if i == 0:
                    continue
                if gripper_state[other_keypoint][0] != gripper_state[i - 1][0]:
                    break

        if verbose:
            print(
                f"reference_pt: {reference_pt}, min_gripper: {min_gripper}, prop: {proprio}"
            )

        # choose an input frame-idx, in our case this is the 1st frame
        # associated with current keypoint
        input_keyframes = self.extract_manual_keyframes(
            trial["input_keyframe"][()])
        if self.use_first_frame_as_input:
            if verbose:
                print(
                    "use_first_frame_as_input was used but it doesn't do anything right now"
                )
        #     image_index = keypoint_idx % 1
        #     # use one of the 1st two frames
        #     # TODO replace this with bursts of images around each keyframe which we can sample from
        #     # helps with overindexing on gripper/other unmoving objects
        # input_idx = np.concatenate([np.zeros(1), keypoints])[chosen_idx].astype(int)
        if chosen_idx == 2:
            input_idx = k_idx - 1
        else:
            input_idx = keypoints[0]
        if verbose:
            # print(f"Chosen image index: {image_index}")
            print(f"Input index for proprio: {input_idx}")

        time_step = np.array([(chosen_idx / (len(keypoints) - 1) - 0.5) * 2])
        proprio = np.concatenate(
            (gripper_state[input_idx], proprio[input_idx], time_step)
        )
        if verbose:
            print(f"Proprio: {proprio}")

        # get point-cloud in base-frame from the cameras
        rgbs, xyzs = [], []
        # print(f"image_index: {image_index}")
        for view in ["wrist"]:
            for image_index in input_keyframes:
                v_rgb, v_xyz = self.process_images_from_view(
                    trial,
                    view,
                    image_index if image_index is not None else input_idx,
                )
                rgbs.append(v_rgb)
                xyzs.append(v_xyz)
        rgb = np.concatenate(rgbs, axis=0)
        xyz = np.concatenate(xyzs, axis=0)
        z_mask = xyz[:, 2] > 0.0
        rgb = rgb[z_mask]
        xyz = xyz[z_mask]
        x_mask = xyz[:, 0] < 0.9
        rgb = rgb[x_mask]
        xyz = xyz[x_mask]

        # Get only a few points that we care about here
        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)

        # get EE keyframe
        ee_keyframe = self.get_gripper_pose(trial, int(k_idx))
        ref_ee_keyframe = self.get_gripper_pose(trial, int(reference_pt))
        keyframes = []
        if self.multi_step:
            target_gripper_state = np.zeros(len(keypoints))
            # Add all keypoints to this list
            for j, keypoint in enumerate(keypoints):
                keyframes.append(self.get_gripper_pose(trial, int(keypoint)))
                target_gripper_state[j] = gripper_state[keypoint]
        else:
            # Pull out gripper state from the sim data
            target_gripper_state = gripper_state[k_idx]

        # preserve the og pcd
        og_xyz = xyz.copy()
        og_rgb = rgb.copy()
        # voxelize at a granular voxel-size then choose X points
        xyz, rgb = self.remove_duplicate_points(xyz, rgb)
        xyz, rgb = self.dr_crop_radius(xyz, rgb, ref_ee_keyframe)
        orig_xyz, orig_rgb = xyz, rgb

        # Get the point clouds and shuffle them around a bit
        xyz, rgb, center = self.downsample_point_cloud(xyz, rgb)

        # mean-center the keyframes wrt classifier-input pcd
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
        ) = self.get_query(xyz, rgb, ref_ee_keyframe)
        if xyz2 is None:
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
            crop_location, ee_keyframe, ref_ee_keyframe, keyframes
        )

        # Debug code
        # TODO - remove this debug code
        # pos1 = crop_ee_keyframe[:3, 3]
        # pos2 = np.linalg.inv(crop_ee_keyframe)[:3, 3]
        # print()
        # print(cmd, pos1, pos2)

        # Get the commands we care about here
        # TODO - remove debug code
        # print(crop_ee_keyframe[:3, 3])
        positions, orientations, angles = self.get_commands(
            crop_ee_keyframe, crop_keyframes
        )
        self._assert_positions_match_ee_keyframes(crop_ee_keyframe, positions)

        if self.visualize:
            show_point_cloud_with_keypt_and_closest_pt(
                xyz,
                rgb,
                ee_keyframe[:3, 3],
                ee_keyframe[:3, :3],
                ref_ee_keyframe[:3, 3],
            )
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                ee_keyframe[:3, 3],
                ee_keyframe[:3, :3],
                ref_ee_keyframe[:3, 3],
            )
            # show_point_cloud_with_keypt_and_closest_pt(
            #     crop_xyz,
            #     crop_rgb,
            #     crop_ee_keyframe[:3, 3],
            #     crop_ee_keyframe[:3, :3],
            #     None,
            # )

        datum = {
            "trial_name": trial.name,
            "data_ok_status": data_status,
            # ----------
            "ee_keyframe_pos": torch.FloatTensor(ee_keyframe[:3, 3]),
            "ee_keyframe_ori": torch.FloatTensor(ee_keyframe[:3, :3]),
            "proprio": torch.FloatTensor(proprio),
            "target_gripper_state": torch.FloatTensor(target_gripper_state),
            "xyz": torch.FloatTensor(xyz),
            "rgb": torch.FloatTensor(rgb),
            "cmd": cmd,
            "keypoint_idx": keypoint_idx,
            # engineered features ----------------
            "closest_pos": torch.FloatTensor(closest_pt_og_pcd),
            "closest_pos_idx": torch.LongTensor([target_idx_og_pcd]),
            "closest_voxel": torch.FloatTensor(closest_pt_down_pcd),
            "closest_voxel_idx": torch.LongTensor([target_idx_down_pcd]),
            "xyz_downsampled": torch.FloatTensor(xyz2),
            "rgb_downsampled": torch.FloatTensor(rgb2),
            # used in pt_query.py; make sure this is being used with xyz_downsampled
            # TODO rename xyz_mask --> xyz_downsampled_mask to remove confusion
            "xyz_mask": torch.LongTensor(self.mask_voxels(xyz2, target_idx_down_pcd)),
            # Crop inputs -----------------
            "rgb_crop": torch.FloatTensor(crop_rgb),
            "xyz_crop": torch.FloatTensor(crop_xyz),
            "crop_ref_ee_keyframe_pos": torch.FloatTensor(crop_ref_ee_keyframe[:3, 3]),
            "crop_ref_ee_keyframe_ori": torch.FloatTensor(crop_ref_ee_keyframe[:3, :3]),
            # Crop goals ------------------
            # Goals for regression go here
            "ee_keyframe_pos_crop": torch.FloatTensor(positions),
            "ee_keyframe_ori_crop": torch.FloatTensor(orientations),
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
@click.option("-ki", "--k-index", default=0)
def debug_get_datum(data_dir, k_index, split):
    with open(split, "r") as f:
        train_test_split = json.load(f)
    # debug_list = ["26_11_2022_18_40_48", "26_11_2022_18_43_08"]
    loader = RobotDataset(
        data_dir,
        num_pts=8000,
        data_augmentation=True,
        random_idx=False,
        ori_dr_range=np.pi / 8,
        first_frame_only=True,
        # first_keypoint_only=True,
        keypoint_range=[k_index],
        trial_list=train_test_split["test"],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=True,
        show_cropped=True,
        verbose=True,
    )
    for trial in loader.trials:
        if "bottom" in trial.h5_filename:
            print(f"Trial name: {trial.name}")
            data = loader.get_datum(trial, k_index)


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default="/home/priparashar/robopen_h5s/larp/9tasks_woutclutter",
)
@click.option("--split", help="json file with train-test-val split")
@click.option("--template", default="*.h5")
def show_all_keypoints(data_dir, split, template):
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
        data_augmentation=False,
        crop_radius=True,
        random_idx=False,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_only=True,
        # first_keypoint_only=True,
        keypoint_range=[0, 1, 2],
        trial_list=train_test_split["train"],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=True,
        show_cropped=True,
        verbose=False,
        multi_step=False,
        visualize=True,
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
                data = loader.get_datum(trial, i, verbose=False)
            # data = loader.get_datum(trial, 1, verbose=False)


if __name__ == "__main__":
    show_all_keypoints()
    pass

