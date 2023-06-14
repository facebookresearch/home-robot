import json
import unittest
from typing import Optional

import click
import clip
import matplotlib.pyplot as plt
import numpy as np
import slap_manipulation.policy.peract_utils as utils

# import peract_colab.arm.utils as utils
import torch
import yaml
from slap_manipulation.dataloaders.robot_loader import RobotDataset
from slap_manipulation.policy.voxel_grid import VoxelGrid


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
            -0.5,
            -0.2,
            -0.5,
            0.5,
            0.8,
            0.5,
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

            # add language signals
            new_data["cmd"] = data["cmd"]
            description = data["cmd"]
            new_data["gripper_width_array"] = data["gripper_width_array"]
            new_data["all_proprio"] = data["all_proprio"]
            # tokens = clip.tokenize([description]).numpy()
            # token_tensor = torch.from_numpy(tokens).to(self.device)
            # lang, lang_emb = self._clip_encode_text(token_tensor)
            # lang = self.clip_encode_text(description)
            # new_data["cmd_embs"] = lang

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

    def _norm_rgb(self, x):
        return (x.float() / 1.0) * 2.0 - 1.0

    def _preprocess_inputs(self, sample):
        rgb = sample["rgb"]
        pcd = sample["xyz"]

        rgb = self._norm_rgb(rgb)

        # not required as we are not using the baselines below peract
        # obs.append(
        #     [rgb, pcd]
        # )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        return rgb, pcd

    def is_action_valid(self, sample):
        """
        simple method which compares ee-keyframe position with the scene bounds
        returns False if action is outside the scene bounds
        """
        bounds = np.array(self.scene_bounds)
        if (sample["ee_keyframe_pos"].numpy() > bounds[..., 3:]).any() or (
            sample["ee_keyframe_pos"].numpy() < bounds[..., :3]
        ).any():
            return False
        return True

    # discretize translation, rotation, gripper open, and ignore collision actions
    def _get_action(
        self,
        sample: dict,
        rotation_resolution: int,
        idx: int = -1,
        max_keypoints=6,
    ):
        if idx == -1:
            quat = sample["ee_keyframe_ori_quat"]
            attention_coordinate = sample["ee_keyframe_pos"]
            grip = float(sample["proprio"][-1])
            D = len(self.voxel_sizes)
            gripper_width = sample["gripper_width_array"]
        else:
            quat = sample["ee_keyframe_ori_quat"][idx]
            attention_coordinate = sample["ee_keyframe_pos"][idx]
            grip = float(sample["all_proprio"][idx, -1])
            # gripper_state = sample["target_gripper_state"][idx]
            # grip = float(sample["proprio"][idx][-1])
            gripper_width = sample["gripper_width_array"][idx]
            time_index = float((2.0 * idx) / max_keypoints)
        quat = utils.normalize_quaternion(quat)
        if quat[-1] < 0:
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
            torch.cat((gripper_width, torch.FloatTensor([grip, time_index]))),
            torch.Tensor(attention_coordinate).unsqueeze(0),
        )

    # extract CLIP language features for goal string
    def _clip_encode_text(self, text):
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

    def clip_encode_text(self, text):
        """encode text as a sequence"""

        with torch.no_grad():
            lang = clip.tokenize(text).to(self.device)
            lang = self.clip_model.token_embedding(lang).type(
                self.clip_model.dtype
            ) + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            lang = lang.permute(1, 0, 2)
            lang = self.clip_model.transformer(lang)
            lang = lang.permute(1, 0, 2)
            lang = self.clip_model.ln_final(lang).type(self.clip_model.dtype)

        # We now have per-word clip embeddings
        lang = lang.float()

        # Encode language here
        batch_size, lang_seq_len, _ = lang.shape
        # lang = lang.view(batch_size * lang_seq_len, -1)
        # learned pos encodings will be added in PerAct
        # if self.learned_pos_encoding:
        #     lang = self.lang_preprocess(lang) + self.pos_encoding
        # else:
        #     lang = self.lang_preprocess(lang)
        #     lang = self.pos_encoding(lang)
        lang = lang.view(batch_size, lang_seq_len, -1)

        return lang

    def visualize_data(self, batch, vox_grid=None):
        if vox_grid is None:
            vox_grid = self.vox_grid
        lang_goal = batch["cmd"]
        batch = {
            k: v.to(self.device) for k, v in batch.items() if type(v) == torch.Tensor
        }

        # get obs
        flat_img_features = batch["rgb"]
        pcd_flat = batch["xyz"]

        # flatten observations
        # bs = obs[0][0].shape[0]
        # pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)

        # image_features = [o[0] for o in obs]
        # feat_size = image_features[0].shape[1]
        # flat_imag_features = torch.cat(
        #     [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)

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
            plt.pause(2)


class TestPerActDataloader(unittest.TestCase):
    # def test_dataloader(self):
    #     # TODO convert split file from json to YAML
    #     data_dir = "/home/priparashar/robopen_h5s/larp/unit_test_pick_bottle"
    #     split_file = "/home/priparashar/Development/icra/home_robot/assets/train_test_val_split_9tasks_2022-12-01.json"
    #     with open(split_file, "r") as f:
    #         train_test_split = json.load(f)
    #     ds = PerActRobotDataset(
    #         data_dir,
    #         trial_list=train_test_split["train"],
    #     )
    #     # Create data loaders
    #     num_workers = 0
    #     B = 1
    #     data_loader = torch.utils.data.DataLoader(
    #         ds,
    #         batch_size=B,
    #         num_workers=num_workers,
    #         shuffle=True,
    #         drop_last=True,
    #     )
    #     for batch in data_loader:
    #         ds.visualize_data(batch)

    def test_is_action_valid(self):
        data_dir = "/home/priparashar/robopen_h5s/larp/unit_test_pick_bottle"
        split_file = "/home/priparashar/Development/icra/home_robot/assets/train_test_val_split_9tasks_2022-12-01.json"
        with open(split_file, "r") as f:
            train_test_split = json.load(f)
        ds = PerActRobotDataset(
            data_dir,
            trial_list=train_test_split["train"],
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
    loader = PerActRobotDataset(
        data_dir,
        template="*.h5",
        num_pts=8000,
        data_augmentation=False,
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
        per_action_cmd=False,
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
    ds = PerActRobotDataset(
        data_dir,
        template=template,
        num_pts=8000,
        data_augmentation=False,
        crop_radius=True,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        # first_keypoint_only=True,
        # keypoint_range=[0],
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
        ds.visualize_data(batch)
    # skip_names = ["30_11_2022_15_22_40"]
    # for trial in loader.trials:
    #     print(f"Trial: {trial.name}")
    #     print(f"Task name: {trial.h5_filename}")
    #     if trial.name in skip_names:
    #         print("skipping as known bad trajectory")
    #     else:
    #         num_keypt = trial.num_keypoints
    #         for i in range(num_keypt):
    #             print("Keypoint requested: ", i)
    #             data = loader.get_datum(trial, i, verbose=True)
    #             breakpoint()
    #         # data = loader.get_datum(trial, 1, verbose=False)


if __name__ == "__main__":
    show_all_keypoints()
    pass
# def main():
#     # TODO convert split file from json to YAML
#     data_dir = "/home/priparashar/robopen_h5s/larp/unit_test_pick_bottle"
#     split_file = "/home/priparashar/Development/icra/home_robot/assets/train_test_val_split_9tasks_2022-12-01.json"
#     with open(split_file, "r") as f:
#         train_test_split = json.load(f)
#     ds = PerActRobotDataset(
#         data_dir, trial_list=train_test_split["train"], num_pts=8000
#     )
#     # Create data loaders
#     num_workers = 0
#     B = 1
#     data_loader = torch.utils.data.DataLoader(
#         ds,
#         batch_size=B,
#         num_workers=num_workers,
#         shuffle=True,
#         drop_last=True,
#     )
#     for batch in data_loader:
#         ds.visualize_data(batch)
#
#
# if __name__ == "__main__":
#     main()
