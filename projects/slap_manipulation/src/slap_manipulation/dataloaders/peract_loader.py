# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import time
import unittest
from typing import Any, Dict, Optional

import click
import clip
import matplotlib.pyplot as plt
import numpy as np
import slap_manipulation.policy.peract_utils as utils

# import peract_colab.arm.utils as utils
import torch
import trimesh.transformations as tra
import yaml
from slap_manipulation.dataloaders.robot_loader import RobotDataset
from slap_manipulation.policy.voxel_grid import VoxelGrid

from home_robot.utils.point_cloud import show_point_cloud


class PerActRobotDataset(RobotDataset):
    """
    dataloader wrapping default dataloader for SLAP
    """

    def __init__(
        self,
        dirname,
        template="*.h5",
        verbose=False,
        num_pts=8000,
        data_augmentation=True,
        random_cmd=True,
        first_keypoint_only=False,
        keypoint_range: Optional[list] = None,
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
        query_radius: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(
            dirname,
            template=template,
            num_pts=num_pts,
            data_augmentation=data_augmentation,
            random_cmd=random_cmd,
            first_keypoint_only=first_keypoint_only,
            keypoint_range=keypoint_range,
            show_voxelized_input_and_reference=show_voxelized_input_and_reference,
            show_raw_input_and_reference=show_raw_input_and_reference,
            show_cropped=show_cropped,
            ori_dr_range=ori_dr_range,
            cart_dr_range=cart_dr_range,
            first_frame_as_input=first_frame_as_input,
            trial_list=trial_list,
            orientation_type=orientation_type,
            multi_step=multi_step,
            crop_radius=crop_radius,
            ambiguous_radius=ambiguous_radius,
            crop_radius_chance=crop_radius_chance,
            crop_radius_shift=crop_radius_shift,
            crop_radius_range=crop_radius_range,
            visualize_interaction_estimates=visualize_interaction_estimates,
            visualize_cropped_keyframes=visualize_cropped_keyframes,
            yaml_file=yaml_file,  # "./assets/language_variations/v0.yml",
            robot=robot,
            depth_factor=depth_factor,
            autoregressive=autoregressive,
            max_keypoints=max_keypoints,
            time_as_one_hot=time_as_one_hot,
            per_action_cmd=per_action_cmd,
            skill_to_action_file=skill_to_action_file,
            query_radius=query_radius,
        )

        # add a default device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            self.clip_model, self.preprocess = clip.load(
                # "RN50", device=self.device  # network used in peract
                "ViT-B/32",
                device=self.device,  # network used in SLAP
            )
        # scene bounds used to voxelize a part of the scene
        self.scene_bounds = [
            -0.75,
            -0.75,
            -0.75,
            0.75,
            0.75,
            0.75,
        ]
        self._retries = 10
        # [x_min,y_min,z_min,x_max,y_max,z_max] - metric volume to be voxelized
        self.image_width = 640
        self.image_height = 480
        self.cameras = ["head"]
        self.batch_size = 1
        self.voxel_sizes = [100]
        self.num_pts = num_pts
        # initialize voxelizer
        self.vox_grid = VoxelGrid(
            coord_bounds=self.scene_bounds,
            voxel_size=self.voxel_sizes[0],
            device=self.device,
            batch_size=1,
            feature_size=3,
            max_num_coords=self.num_pts,  # self.num_pts
        )

    def get_datum(self, trial, keypoint_idx, verbose=False):
        tries = 0
        while tries < self._retries:
            new_data = {}
            data = super().get_datum(trial, keypoint_idx, verbose=verbose)
            if not data["data_ok_status"]:
                tries += 1
                continue
            else:
                new_data["data_ok_status"] = True

            # add new signals needed
            # NOTE loader returns trimesh quat, i.e. w,x,y,z
            # peract expects scipy quat x,y,z,w
            all_keypoints = data["global_proprio"]
            new_data["ee_keyframe_ori_quat"] = torch.roll(all_keypoints[:, 3:7], 1, -1)
            new_data["ee_keyframe_pos"] = all_keypoints[:, :3]
            new_data["proprio"] = data["all_proprio"]

            # add language  and proprio signals
            new_data["cmd"] = data["cmd"]
            new_data["all_proprio"] = data["all_proprio"]
            new_data["num_actions"] = data["num_keypoints"]
            new_data["peract_input"] = data["peract_input"]

            # discretize supervision signals
            (
                trans_action_indices,
                grip_rot_action_indices,
                ignore_colls,
                gripper_states,
                attention_coords,
            ) = ([], [], [], [], [])
            for idx in range(len(data["global_proprio"])):
                (
                    trans_action_index,
                    grip_rot_action_index,
                    ignore_coll,
                    gripper_state,
                    attention_coord,
                ) = self._get_action(new_data, 5, idx, len(data["global_proprio"]))
                trans_action_indices.append(trans_action_index)
                grip_rot_action_indices.append(grip_rot_action_index)
                ignore_colls.append(ignore_coll)
                gripper_states.append(gripper_state.unsqueeze(0))
                attention_coords.append(attention_coord)
            trans_action_indices = torch.cat(trans_action_indices)
            grip_rot_action_indices = torch.cat(grip_rot_action_indices)
            ignore_colls = torch.cat(ignore_colls)
            gripper_states = torch.cat(gripper_states, dim=0)
            attention_coords = torch.cat(attention_coords)

            # # add discretized signal to dictionary
            # # concatenate gripper ori and pos
            gripper_pose = torch.FloatTensor(data["global_proprio"][:, :-1])
            new_data.update(
                {
                    "trans_action_indices": trans_action_indices,
                    "rot_grip_action_indices": grip_rot_action_indices,
                    "ignore_collisions": ignore_colls,
                    "gripper_pose": gripper_pose,
                    "gripper_states": gripper_states,
                    "attention_coords": attention_coords,
                }
            )

            # TODO make sure you update the keys used in peract_model.update() method
            # preprocess and normalize obs
            rgb, pcd = self._preprocess_inputs(data)
            new_data["rgb"] = rgb
            new_data["xyz"] = pcd
            new_data["action"] = gripper_states

            if self.is_action_valid(new_data):
                return new_data
            else:
                tries += 1

        return new_data

    def get_per_waypoint_batch(self, batch: Dict[str, Any], idx: int):
        """Used on the batch received from dataloader to extract input and output for a single waypoint
        :batch: batch received from dataloader
        :idx: index of the waypoint
        """
        new_data = {"batch": batch, "idx": idx}

        # This gives us our action translation indices - location in the voxel cube
        action_trans = batch["trans_action_indices"][:, :, :3].int()
        # Rotation index
        action_rot_grip = batch["rot_grip_action_indices"][:, :].int()
        # Do we take some action to ignore collisions or not
        action_ignore_collisions = batch["ignore_collisions"][:, -1].int()

        # Get language goal embedding
        lang_goal = batch["cmd"]

        obs = batch["rgb"]
        pcd = batch["xyz"]

        # inputs
        # 3 dimensions - proprioception: {gripper_open, gripper_width, timestep}
        proprio = batch["gripper_states"]
        proprio_instance = proprio[:, idx]
        action_trans_instance = action_trans[:, idx]
        action_rot_grip_instance = action_rot_grip[:, idx]
        action_ignore_collisions_instance = action_ignore_collisions

        new_data["proprio"] = proprio_instance

        new_data["action_trans"] = action_trans_instance
        new_data["action_rot_grip"] = action_rot_grip_instance
        new_data["action_ignore_collisions"] = action_ignore_collisions_instance
        new_data["cmd"] = lang_goal
        new_data["rgb"] = obs
        new_data["xyz"] = pcd
        new_data["target_continuous_position"] = new_data["batch"]["ee_keyframe_pos"][
            :, idx
        ]
        new_data["target_continuous_orientation"] = new_data["batch"][
            "ee_keyframe_ori_quat"
        ][:, idx]
        return new_data

    def _norm_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB images to [-1, 1]"""
        return (x.float() / 1.0) * 2.0 - 1.0

    def _preprocess_inputs(self, sample: Dict[str, Any]):
        """Preprocess inputs for peract model; by normalizing RGB b/w -1 to 1
        :sample: the batch received from dataloader"""
        rgb = sample["rgb"]
        pcd = sample["xyz"]

        rgb = self._norm_rgb(rgb)

        return rgb, pcd

    def is_action_valid(self, sample: Dict[str, Any]):
        """
        PerAct can only process Q-values for an action if the action falls within
        it's scene-bounds (defined in the config file, used to create VoxelGrid)
        This method makes sure the action is learnable wrt PerAct's scene bounds
        returns False if action is outside the scene bounds
        :sample: batch received from DataLoader which has the action
        """
        bounds = np.array(self.scene_bounds)
        if (sample["ee_keyframe_pos"].numpy() > bounds[..., 3:]).any() or (
            sample["ee_keyframe_pos"].numpy() < bounds[..., :3]
        ).any():
            return False
        return True

    def _get_action(
        self,
        sample: Dict[str, Any],
        rotation_resolution: int,
        idx: int = -1,
        max_keypoints: int = 6,
    ):
        """discretize translation, rotation, gripper open, and ignore collision actions
        :sample: batch received from dataloader
        :rotation_resolution: number of discrete rotation actions
        :idx: index of the action needed to be discretized (i.e. from trajectory
        - )
        :max_keypoints: max number of keypoints supported in the trajectory (used for calculating time-index)
        """
        if idx == -1:
            quat = sample["ee_keyframe_ori_quat"]
            attention_coordinate = sample["ee_keyframe_pos"]
            grip = float(sample["proprio"][-1])
            D = len(self.voxel_sizes)
            gripper_width = sample["peract_input"]["gripper_width_array"]
        else:
            quat = sample["ee_keyframe_ori_quat"][idx]
            attention_coordinate = sample["ee_keyframe_pos"][idx]
            grip = sample["peract_input"]["gripper_action"][idx].float()
            gripper_width = sample["peract_input"]["gripper_width"][idx]
            time_index = float((2.0 * idx) / max_keypoints - 1)
        quat = utils.normalize_quaternion(quat)
        if (
            quat[-1] < 0
        ):  # so quaternions map to consistent rotation bin after discretization
            quat = -quat
        disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
        trans_indices, attention_coordinates = [], []
        bounds = np.array(self.scene_bounds)
        for depth, vox_size in enumerate(
            self.voxel_sizes
        ):  # only single voxelization-level is used in PerAct
            # all of the following args should be numpy
            index = utils.point_to_voxel_index(
                attention_coordinate.numpy(), vox_size, bounds
            )
            trans_indices.extend(index.tolist())
            res = (bounds[3:] - bounds[:3]) / vox_size
            attention_coordinate = bounds[:3] + res * index
            attention_coordinates.append(attention_coordinate)
        rot_and_grip_indices = disc_rot.tolist()
        rot_and_grip_indices.extend([grip])
        ignore_collisions = [int(0.0)]  # hard-code
        return (
            torch.Tensor([trans_indices]),
            torch.Tensor([rot_and_grip_indices]),
            torch.Tensor([ignore_collisions]),
            torch.cat(
                (
                    gripper_width.unsqueeze(0),
                    grip,
                    torch.FloatTensor([time_index]),
                )
            ),
            torch.Tensor(attention_coordinate).unsqueeze(0),
        )

    def _clip_encode_text(self, text: str):
        """extract CLIP language features from :text: for goal string"""
        x = self.clip_model.token_embedding(text).type(
            self.clip_model.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        emb = x.clone()
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.clip_model.text_projection
        )

        return x, emb

    def visualize_data(self, batch, vox_grid=None):
        """visualize the data for debugging
        :batch: data received from dataloader
        :vox_grid: optional, external voxel_grid to be used for visualizing this
        """
        if vox_grid is None:
            vox_grid = self.vox_grid
        lang_goal = batch["cmd"]
        batch = {
            k: v.to(self.device) for k, v in batch.items() if type(v) == torch.Tensor
        }

        # get obs
        flat_img_features = batch["rgb"]
        pcd_flat = batch["xyz"]

        # tensorize scene bounds
        bounds = torch.tensor(self.scene_bounds, device=self.device).unsqueeze(0)

        # voxelize!
        voxel_grid = vox_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_img_features, coord_bounds=bounds
        )

        # swap to channels fist
        vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()

        # expert action voxel indicies
        print(f"Lang goal: {lang_goal}")
        vis_gt_coord = None
        for idx in range(batch["trans_action_indices"].shape[1]):
            vis_gt_coord = (
                batch["trans_action_indices"][:, idx, :3].int().detach().cpu().numpy()
            )

            # render voxel grid with expert action (blue)
            # Show voxel grid and expert action (blue)
            rotation_amount = 45
            rendered_img = utils.visualise_voxel(
                vis_voxel_grid[0],
                None,
                None,
                vis_gt_coord[0],
                voxel_size=0.045,
                rotation_amount=np.deg2rad(rotation_amount),
            )

            fig = plt.figure(figsize=(15, 15))
            plt.imshow(rendered_img)
            plt.axis("off")
            plt.pause(3)
            plt.close()


class TestPerActDataloader(unittest.TestCase):
    def test_is_action_valid(self):
        data_dir = "./data/"
        ds = PerActRobotDataset(
            data_dir,
            trial_list=[],
        )
        fake_data = {"ee_keyframe_pos": torch.Tensor([-9.0, -9.0, -9.0])}
        self.assertFalse(ds.is_action_valid(fake_data))
        fake_data = {"ee_keyframe_pos": torch.Tensor([9.0, 9.0, 9.0])}
        self.assertFalse(ds.is_action_valid(fake_data))
        fake_data = {"ee_keyframe_pos": torch.Tensor([0.0, 0.0, 0.0])}
        self.assertTrue(ds.is_action_valid(fake_data))


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default="./data/",
)
@click.option("--split", help="json file with train-test-val split")
@click.option(
    "--waypoint-language",
    help="yaml for skill-to-action lang breakdown",
    default="",
)
@click.option("-ki", "--k-index", default=[0], multiple=True)
@click.option("-r", "--robot", default="stretch")
def debug_get_datum(data_dir, k_index, split, robot, waypoint_language):
    if split:
        with open(split, "r") as f:
            train_test_split = yaml.safe_load(f)
    loader = PerActRobotDataset(
        data_dir,
        template="*.h5",
        num_pts=8000,
        data_augmentation=True,
        crop_radius=True,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        trial_list=train_test_split["train"] if split else [],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=False,
        show_cropped=False,
        verbose=False,
        multi_step=True,
        visualize_interaction_estimates=False,
        visualize_cropped_keyframes=False,
        robot=robot,
        autoregressive=True,
        time_as_one_hot=True,
        per_action_cmd=False,
        skill_to_action_file=None if waypoint_language == "" else waypoint_language,
    )
    for trial in loader.trials:
        print(f"Trial name: {trial.name}")
        for k_i in k_index:
            data = loader.get_datum(trial, k_i, verbose=False)


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default="./data",
)
@click.option("--split", help="json file with train-test-val split")
@click.option("--template", default="*.h5")
@click.option("--robot", default="stretch")
@click.option("--test-pose/--no-test-pose", default=False)
@click.option("--test-voxel-grid/--no-test-voxel-grid", default=False)
def show_all_keypoints(data_dir, split, template, robot, test_pose, test_voxel_grid):
    """function which visualizes keypoints overlaid on initial frame, then
    visualizes the input frame for each keypoint with labeled interaction
    point overlaid"""
    if test_pose and test_voxel_grid:
        print("You can only test Poses or VoxelGrid at a time")
        return
    if split:
        with open(split, "r") as f:
            train_test_split = yaml.safe_load(f)
    ds = PerActRobotDataset(
        data_dir,
        template=template,
        num_pts=8000,
        data_augmentation=True,
        crop_radius=False,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        trial_list=train_test_split["train"] if split else [],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=False,
        show_cropped=False,
        verbose=False,
        visualize_interaction_estimates=False,
        visualize_cropped_keyframes=False,
        robot="stretch",
        autoregressive=True,
        time_as_one_hot=True,
        multi_step=True,
    )
    # Create data loaders
    num_workers = 0
    B = 1
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=B,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    for batch in data_loader:
        if test_voxel_grid:
            ds.visualize_data(batch)
        if test_pose:
            num_actions = batch["num_actions"][0]
            for i in range(num_actions):
                wp_batch = ds.get_per_waypoint_batch(batch, i)
                xyz = wp_batch["xyz"][0]
                rgb = wp_batch["rgb"][0]
                # NOTE prior position, orientation does not matter to peract
                # get input gripper-state
                print("Task:", wp_batch["cmd"][0])
                print("Input gripper state:", wp_batch["proprio"][0])
                # get target position
                position = wp_batch["target_continuous_position"]
                # get target orientation
                orientation = wp_batch["target_continuous_orientation"]
                print("position:", position, "; orientation:", orientation)
                show_point_cloud(
                    xyz,
                    rgb,
                    orig=position.reshape(3, 1),
                    R=tra.quaternion_matrix(np.roll(orientation, -1))[:3, :3],
                )


if __name__ == "__main__":
    show_all_keypoints()
