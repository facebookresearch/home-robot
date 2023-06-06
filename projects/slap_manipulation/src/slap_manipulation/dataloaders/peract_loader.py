import json
import unittest

import click
import clip
import matplotlib.pyplot as plt
import numpy as np
import peract_colab.arm.utils as utils
import torch
from arm.c2farm.voxel_grid import VoxelGrid
from peract_colab.arm.utils import (
    discrete_euler_to_quaternion,
    get_gripper_render_pose,
    visualise_voxel,
)

from home_robot.datasets.robopen_loader import RoboPenDataset


class PerActRobotDataset(RoboPenDataset):
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
            -0.5,
            -0.5,
            0.5,
            0.5,
            0.5,
        ]
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
        new_data = {}
        data = super().get_datum(trial, keypoint_idx, verbose=verbose)
        if not data["data_ok_status"]:
            return data
        else:
            new_data["data_ok_status"] = True

        # add new signals needed
        # NOTE loader returns trimesh quat, i.e. w,x,y,z
        # peract expects scipy quat x,y,z,w
        new_data["ee_keyframe_ori_quat"] = torch.roll(data["target_ee_angles"], 1, -1)
        new_data["ee_keyframe_pos"] = data["ee_keyframe_pos"]
        new_data["proprio"] = data["proprio"]

        # add language signals
        new_data["cmd"] = data["cmd"]
        description = data["cmd"]
        # tokens = clip.tokenize([description]).numpy()
        # token_tensor = torch.from_numpy(tokens).to(self.device)
        # lang, lang_emb = self._clip_encode_text(token_tensor)
        # lang = self.clip_encode_text(description)
        # new_data["cmd_embs"] = lang

        # discretize supervision signals
        (
            trans_action_indices,
            grip_rot_action_indices,
            ignore_coll,
            gripper_state,
            attention_coords,
        ) = self._get_action(data, 5)

        # add discretized signal to dictionary
        # concatenate gripper ori and pos
        gripper_pose = torch.cat(
            (data["ee_keyframe_pos"], data["target_ee_angles"]), dim=-1
        ).unsqueeze(0)
        new_data.update(
            {
                "trans_action_indices": trans_action_indices,
                "rot_grip_action_indices": grip_rot_action_indices,
                "ignore_collisions": ignore_coll,
                "gripper_pose": gripper_pose,
                "attention_coords": attention_coords,
            }
        )

        # TODO make sure you update the keys used in peract_model.update() method
        # preprocess and normalize obs
        rgb, pcd = self._preprocess_inputs(data)
        new_data["rgb"] = rgb
        new_data["xyz"] = pcd
        new_data["action"] = gripper_state

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
    ):
        quat = utils.normalize_quaternion(sample["target_ee_angles"])
        if quat[-1] < 0:
            quat = -quat
        disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
        attention_coordinate = sample["ee_keyframe_pos"]
        trans_indices, attention_coordinates = [], []
        bounds = np.array(self.scene_bounds)
        ignore_collisions = [int(0.0)]  # hard-code
        for depth, vox_size in enumerate(
            self.voxel_sizes
        ):  # only single voxelization-level is used in PerAct
            # all of the following args should be numpy
            index = utils.point_to_voxel_index(
                sample["ee_keyframe_pos"].numpy(), vox_size, bounds
            )
            trans_indices.extend(index.tolist())
            res = (bounds[3:] - bounds[:3]) / vox_size
            attention_coordinate = bounds[:3] + res * index
            attention_coordinates.append(attention_coordinate)

        rot_and_grip_indices = disc_rot.tolist()
        grip = float(sample["proprio"][1])
        rot_and_grip_indices.extend([int(sample["target_gripper_state"])])
        D = len(self.voxel_sizes)
        return (
            torch.Tensor([trans_indices]),
            torch.Tensor([rot_and_grip_indices]),
            torch.Tensor([ignore_collisions]),
            torch.Tensor(
                np.concatenate([sample["target_gripper_state"], np.array([grip])])
            ),
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
        vis_gt_coord = (
            batch["trans_action_indices"][:, -1, :3].int().detach().cpu().numpy()
        )

        # render voxel grid with expert action (blue)
        # Show voxel grid and expert action (blue)
        rotation_amount = 00
        rendered_img = visualise_voxel(
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

        print(f"Lang goal: {lang_goal}")


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


def main():
    # TODO convert split file from json to YAML
    data_dir = "/home/priparashar/robopen_h5s/larp/unit_test_pick_bottle"
    split_file = "/home/priparashar/Development/icra/home_robot/assets/train_test_val_split_9tasks_2022-12-01.json"
    with open(split_file, "r") as f:
        train_test_split = json.load(f)
    ds = PerActRobotDataset(
        data_dir, trial_list=train_test_split["train"], num_pts=8000
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


if __name__ == "__main__":
    main()
