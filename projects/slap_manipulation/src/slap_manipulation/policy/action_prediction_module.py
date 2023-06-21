# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import json
import os
import random
from pprint import pprint
from time import time
from typing import Any, Dict, List, Tuple

import clip
import hydra
import numpy as np
import open3d as o3d
import torch
import trimesh.transformations as tra
import wandb
import yaml
from omegaconf import OmegaConf
from slap_manipulation.dataloaders.rlbench_loader import RLBenchDataset
from slap_manipulation.dataloaders.robot_loader import RobotDataset
from slap_manipulation.optim.lamb import Lamb
from slap_manipulation.policy.components import DenseBlock, GlobalSAModule
from slap_manipulation.policy.components import PtnetSAModule as SAModule
from slap_manipulation.policy.mdn import MDN, mdn_loss, sample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MLP, Linear
from tqdm import tqdm

# Default debug dataset paths
# from home_robot.policy.pt_query import train_dataset_dir, valid_dataset_dir
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud

np.random.seed(0)
torch.manual_seed(0)

random.seed(0)


def quaternion_distance(q1, q2):
    """get quaternion distance"""
    assert q1.shape == q2.shape
    return 1 - ((q1 * q2).sum(dim=-1) ** 2)


class QueryRegressionHead(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.ori_type = cfg.orientation_type
        self.pos_in_channels = cfg.regression_head.pos_in_channels
        if self.ori_type == "rpy":
            self.ori_in_channels = 3
        elif self.ori_type == "quaternion":
            self.ori_in_channels = 4  # quaternion output
        else:
            raise NotImplementedError(
                "ori type = " + str(self.ori_type) + " not supported"
            )

        self.final_dim = cfg.regression_head.final_dim
        self.pos_mlp = MLP(
            OmegaConf.to_object(cfg.regression_head.pos_mlp),
            dropout=0.0,
            batch_norm=False,
        )
        self.ori_mlp = MLP(
            OmegaConf.to_object(cfg.regression_head.ori_mlp),
            dropout=0.0,
            batch_norm=False,
        )
        self.gripper_mlp = MLP(
            OmegaConf.to_object(cfg.regression_head.gripper_mlp),
            dropout=0.0,
            batch_norm=False,
        )
        self.pos_mdn = MDN(self.final_dim, self.pos_in_channels, 1)
        self.pos_linear = Linear(self.final_dim, self.pos_in_channels)
        self.ori_linear = Linear(self.final_dim, self.ori_in_channels)
        self.gripper_linear = Linear(self.final_dim, 1)  # proprio_emb dim = 512
        self.to_activation = torch.nn.Sigmoid()
        self.use_mdn = cfg.use_mdn

    def forward(self, x, proprio_task_emb):
        """return a single regression head"""
        # pos_emb = torch.relu(self.pos_mlp(x))
        if self.use_mdn:
            pos_sigma, pos_mu, _ = self.pos_mdn(x)
        else:
            delta_ee_pos = self.pos_linear(x)
        gripper = self.gripper_linear(torch.relu(self.gripper_mlp(proprio_task_emb)))

        abs_ee_ori = self.ori_linear(x)
        if self.ori_type == "quaternion":
            abs_ee_ori = abs_ee_ori / abs_ee_ori.norm(dim=-1)

        if self.use_mdn:
            return pos_sigma, pos_mu, abs_ee_ori, self.to_activation(gripper)
        else:
            return delta_ee_pos, abs_ee_ori, self.to_activation(gripper)


class ActionPredictionModule(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # training and setup vars
        self.ori_type = cfg.orientation_type
        self._lr = cfg.learning_rate
        self._optimizer_type = cfg.optim
        self._lambda_weight_l2 = cfg.lambda_weight_l2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.multi_head = cfg.multi_step
        self.num_heads = cfg.num_heads
        self._crop_size = cfg.crop_size
        self._query_radius = cfg.query_radius
        self._k = cfg.k  # default from pyg example
        self.proprio_in_dim = cfg.dims.proprio_in
        self.image_in_dim = cfg.dims.image_in
        self.proprio_out_dim = cfg.dims.proprio_out
        self.hidden_dim = cfg.hidden_dim
        self.use_mdn = cfg.use_mdn

        self.pos_wt = cfg.weights.position
        self.ori_wt = cfg.weights.orientation
        self.gripper_wt = cfg.weights.gripper

        # encoding language
        # learnable positional encoding
        # Unlike eg in peract, this ONLY applies to the language
        lang_emb_dim, lang_max_seq_len = cfg.dims.lang_emb_out, cfg.lang_max_seq_len
        with torch.no_grad():
            self.clip_model, self.clip_preprocess = clip.load(
                cfg.clip_model, device=self.device
            )

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            self.proprio_in_dim,
            self.proprio_out_dim,
            norm=None,
            activation="relu",
        )

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            0.5,  # fps sampling ratio
            0.5 * self._query_radius,
            MLP(
                OmegaConf.to_object(cfg.model.sa1_mlp),
                batch_norm=False,
                dropout=0.0,
            ),
        )
        self.sa2_module = SAModule(
            0.25,  # this is apparently the FPS sampling ratio
            self._query_radius,
            MLP(
                OmegaConf.to_object(cfg.model.sa2_mlp),
                batch_norm=False,
                dropout=0.0,
            ),
        )
        self.sa3_module = GlobalSAModule(
            MLP(
                OmegaConf.to_object(cfg.model.sa3_mlp),
                batch_norm=False,
                dropout=0.0,
            )
        )
        self.proprio_emb = MLP(
            OmegaConf.to_object(cfg.model.proprio_mlp),
            batch_norm=False,
            dropout=0.0,
        )
        self.lang_emb = MLP(
            OmegaConf.to_object(cfg.model.lang_mlp),
            batch_norm=False,
            dropout=0.0,
        )
        self.time_emb = MLP(
            OmegaConf.to_object(cfg.model.time_mlp),
            batch_norm=False,
            dropout=0.0,
        )
        self.x_gru = torch.nn.GRU(1024, self.hidden_dim, 1)
        self.post_process = MLP(
            OmegaConf.to_object(cfg.model.post_process_mlp),
            batch_norm=False,
            dropout=0.0,
            # activation_layer=torch.nn.LeakyReLU()
        )
        self.pre_process = MLP(
            OmegaConf.to_object(cfg.model.pre_process_mlp),
            batch_norm=False,
            dropout=0.0,
            # activation_layer=torch.nn.LeakyReLU()
        )

        self.regression_head = QueryRegressionHead(cfg)
        self.pos_in_channels = cfg.regression_head.pos_in_channels
        self.ori_in_channels = (
            3 if self.ori_type == "rpy" else 4 if self.ori_type == "quaternion" else -1
        )
        # self._regression_heads = torch.nn.Sequential(*self.regression_heads)
        # self.classify_loss = torch.nn.BCEWithLogitsLoss()
        # self.classify_loss = torch.nn.BinaryCrossEntropyLoss()
        self.classify_loss = torch.nn.BCELoss()
        self.name = f"action_predictor_{cfg.name}"
        self.max_iter = cfg.max_iter

        # for visualizations
        self.cam_view = {
            "front": [-0.89795424592554529, 0.047678244807235863, 0.43749852250766141],
            "lookat": [0.33531651482385966, 0.048464899929339826, 0.54704503365806367],
            "up": [0.43890929711345494, 0.024286597087151203, 0.89820308956788786],
            "zoom": 0.43999999999999972,
        }
        self.setup_training()
        if not cfg.validate and not cfg.dry_run:
            if not os.path.exists(self._save_dir):
                os.mkdir(self._save_dir)
        self.start_time = 0.0

    def setup_training(self):
        # get today's date
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = self.name + "_" + date_time
        # append folder name to current working dir
        path = os.path.join(os.getcwd(), folder_name)
        self._save_dir = path

    def get_optimizer(self):
        """optimizer config"""
        if self._optimizer_type == "lamb":
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            optimizer = Lamb(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception(f"Optimizer not supported: {self._optimizer_type}")
        return optimizer

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def get_best_name(self):
        filename = os.path.join(self._save_dir, "best_" + self.name + ".pth")
        return filename

    def smart_save(self, epoch, val_loss, best_val_loss):
        if val_loss < best_val_loss:
            time_elapsed = int((time() - self.start_time) / 60)
            filename = os.path.join(
                self._save_dir,
                self.name + "_%04d" % (epoch) + "_%06d" % (time_elapsed) + ".pth",
            )
            torch.save(self.state_dict(), filename)
            filename = self.get_best_name()
            torch.save(self.state_dict(), filename)
            return val_loss, True
        return best_val_loss, False

    def to_device(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                new_batch[k] = v
            else:
                new_batch[k] = v.to(self.device)
        return new_batch

    def show_pred_and_grnd_truth(
        self,
        xyz,
        rgb,
        pred_keypt_orig,
        pred_keypt_rot,
        closest_pt=None,
        grnd_orig=None,
        grnd_rot=None,
        save=False,
        i=-1,
        epoch=-1,
        viewpt={},
    ):
        if np.any(rgb) > 1:
            rgb = rgb / 255.0
        pcd = numpy_to_pcd(xyz, rgb)
        geoms = [pcd]
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=pred_keypt_orig
        )
        coords = coords.rotate(pred_keypt_rot)
        geoms.append(coords)
        if closest_pt is not None:
            grnd_closest_pt_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.02
            )
            grnd_closest_pt_sphere.translate(closest_pt)
            grnd_closest_pt_sphere.paint_uniform_color([0, 0.706, 1])
            geoms.append(grnd_closest_pt_sphere)
        if grnd_orig is not None:
            grnd_coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=grnd_orig
            )
            grnd_coords = grnd_coords.rotate(grnd_rot)
            grnd_coords.paint_uniform_color([1, 0, 0])
            geoms.append(grnd_coords)
        o3d.visualization.draw_geometries(geoms)
        # , lookat=self.cam_view["lookat"], up=self.cam_view["up"]
        # )
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # for geom in geoms:
        #     vis.add_geometry(geom)
        #     vis.update_geometry(geom)
        # if viewpt:
        #     ctr = vis.get_view_control()
        #     ctr.set_front(viewpt["front"])
        #     ctr.set_lookat(viewpt["lookat"])
        #     ctr.set_up(viewpt["up"])
        #     ctr.set_zoom(viewpt["zoom"])
        # if save:
        #     vis.poll_updates()
        #     vis.update_renderer()
        #     vis.capture_screen_image(
        #         f"/home/robopen08/.larp/{self.name}_{i}_epoch{epoch}.png"
        #     )
        # else:
        #     vis.run()
        # vis.destroy_window()
        # del vis
        # if viewpt:
        #     del ctr

    def predict(
        self,
        # rgb: np.ndarray,
        rgb_crop: np.ndarray,
        # xyz: np.ndarray,
        xyz_crop: np.ndarray,
        proprio: np.ndarray,
        lang: List[str],
        p_i: np.ndarray,
        rgb_down: np.ndarray,
        xyz_down: np.ndarray,
    ) -> Dict[str, Any]:
        data = {}
        # data["rgb"] = rgb
        # data["xyz"] = xyz
        data["rgb_downsampled"] = rgb_down
        data["xyz_downsampled"] = xyz_down
        data["cmd"] = lang
        data["proprio"] = torch.FloatTensor(proprio)
        data["xyz_crop"] = torch.FloatTensor(xyz_crop)
        data["rgb_crop"] = torch.FloatTensor(rgb_crop)
        data["query_pt"] = torch.FloatTensor(p_i)
        return self.show_validation_on_sensor(data)
        # return output

    def forward(self, xyz, rgb, proprio, time_step, cmd, hidden):
        """
        Classifies the most relevant voxel and uses embedding from that voxel to
        regress residuals on position and orientation of the end-effector.

        feat: tuple of (rgb, rgb_downsampled, proprio)
        pos: tuple of (xyz, xyz_downsampled)
            xyz: point-locations corresponding to each feat
        cmd: language annotation of the current task
        """

        # Extract language. This should let us create more interesting things...
        with torch.no_grad():
            lang = clip.tokenize(cmd).to(self.device)
            lang = self.clip_model.encode_text(lang)
        lang_emb = self.lang_emb(lang.float())

        # condense rgb into a single point embedding
        proprio = self.proprio_preprocess(proprio[None])
        proprio_emb = self.proprio_emb(proprio)
        time_emb = self.time_emb(time_step[None])
        # proprio = proprio[None].repeat(rgb.shape[0], 1)
        # in_feat = torch.cat(  #  not used
        #     [rgb, proprio],
        #     dim=1,
        # )
        sa0_out = (
            rgb,
            xyz,
            torch.zeros(rgb.shape[0]).to(self.device).long(),
        )
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = torch.cat([x, lang_emb, proprio_emb, time_emb], dim=-1)
        proprio_task_emb = torch.cat([lang_emb, proprio_emb, time_emb], dim=-1)
        batch_size = x.shape[0]

        # insert a GRU unit here
        x = torch.relu(self.pre_process(x))
        x, hidden = self.x_gru(x, hidden)
        x = torch.relu(self.post_process(x))

        if not self.use_mdn:
            positions = torch.zeros(batch_size, 1, self.pos_in_channels).to(self.device)
        orientations = torch.zeros(batch_size, 1, self.ori_in_channels).to(self.device)
        grippers = torch.zeros(batch_size, 1, 1).to(self.device)

        # Get the full set of outputs
        if self.use_mdn:
            pos_sigma, pos_mu, abs_ee_ori, gripper = self.regression_head(
                x, proprio_task_emb
            )
        else:
            delta_ee_pos, abs_ee_ori, gripper = self.regression_head(
                x, proprio_task_emb
            )
            positions[:, 0] = delta_ee_pos
        orientations[:, 0] = abs_ee_ori
        grippers[:, 0] = gripper

        if self.use_mdn:
            return pos_sigma, pos_mu, orientations, grippers, hidden
        else:
            return positions, orientations, grippers, hidden

    def get_keypoint(self, batch, i):
        """return input for predicting ith keypoint from batch"""
        proprio = batch["all_proprio"][0][i]
        time_step = batch["all_time_step"][0][i]
        cmd = [batch["all_cmd"][i][0]]
        return proprio, time_step, cmd

    def get_targets(self, batch, i):
        """return targets for ith keypoint"""
        pos = batch["target_ee_keyframe_pos_crop"][0][i]
        ori = batch["target_ee_angles"][0][i]
        g = batch["target_gripper_state"][0][i]
        return pos, ori, g

    def do_epoch(self, data_iter, optimizer, train, unbatched=True):
        if train:
            self.train()
        else:
            self.eval()

        steps = 0
        total_loss = 0
        num_samples = 1

        tot_pos_loss = 0
        tot_ori_loss = 0
        tot_gripper_loss = 0
        for _, batch in enumerate(tqdm(data_iter, ncols=50)):
            if not batch["data_ok_status"]:
                continue
            optimizer.zero_grad()
            batch = self.to_device(batch)
            batch_size = 1
            # xyz = batch["xyz"][0]
            # rgb = batch["rgb"][0]
            # proprio = batch["proprio"][0]
            # cmd = batch["cmd"]
            crop_xyz = batch["xyz_crop"][0]
            crop_rgb = batch["rgb_crop"][0]
            perturbed_crop_location = batch["perturbed_crop_location"]
            num_keypoints = batch["num_keypoints"][0]
            # time_step = batch["time_step"][0]

            # extract supervision terms
            # target_ori = batch["ee_keyframe_ori_crop"]
            # query_idx = batch["closest_pos_idx"][0]
            # query_pt = batch["closest_pos"][0]

            # target_gripper_state = batch["target_gripper_state"][0]
            # target_ee_angles = batch["target_ee_angles"][0]

            pos_loss = 0
            ori_loss = 0
            gripper_loss = 0
            if unbatched:
                for t in range(num_keypoints):
                    if t == 0:
                        hidden = torch.zeros(1, self.hidden_dim).to(self.device)

                    proprio, time_step, cmd = self.get_keypoint(batch, t)
                    target_pos, target_ori, target_g = self.get_targets(batch, t)
                    # target_ori_R = tra.quaternion_matrix(target_ori.detach().cpu().numpy())[:3, :3]
                    # show_point_cloud(crop_xyz.detach().cpu().numpy(), crop_rgb.detach().cpu().numpy(), target_pos.detach().cpu().numpy().reshape(3,1), target_ori_R)

                    # Run the predictor - get positions and orientations for the model
                    if self.use_mdn:
                        (
                            pos_sigma,
                            pos_mu,
                            orientation,
                            gripper,
                            hidden,
                        ) = self.forward(
                            crop_xyz, crop_rgb, proprio, time_step, cmd, hidden
                        )
                        target_pos = target_pos.view(num_samples, 3)
                        pos_loss += mdn_loss(pos_sigma, pos_mu, target_pos)
                    else:
                        position, orientation, gripper, hidden = self.forward(
                            crop_xyz, crop_rgb, proprio, time_step, cmd, hidden
                        )
                        position = position.view(1, 3)
                        target_pos = target_pos.view(1, 3)
                        pos_loss += ((position - target_pos) ** 2).sum()
                    orientation = orientation.view(1, 4)
                    target_ori = target_ori.view(1, 4)

                    ori_loss += quaternion_distance(orientation, target_ori).sum()
                    gripper_loss += self.classify_loss(
                        gripper.view(-1), target_g.view(-1)
                    )
            else:
                hidden = torch.zeros(1, self.hidden_dim).to(self.device)
                # get batched input and targets
                proprio = batch["all_proprio"]
                time_step = batch["all_time_step"]
                cmd = batch["all_cmd"]
                print(proprio)
                print(f"proprio.shape: {proprio.shape}")
                print(time_step)
                print(f"time_step.shape: {time_step.shape}")
                print(cmd)
                print(f"cmd.shape: {cmd.shape}")
                breakpoint()

                # reshape targets

                # process batched sampled
                position, orientation, gripper, hidden = self.forward(
                    crop_xyz, crop_rgb, proprio, time_step, cmd, hidden
                )

                # reshape output
                position = position.view(self.batch_size, 3)
                orientation = orientation.view(self.batch_size, 4)
                gripper = gripper.view(self.batch_size, 1)

            # Compute the position error
            # TODO: what should it be?
            # if self.multi_head:
            #     target_pos = batch["ee_keyframe_pos_crop"]
            #     pred_ee_pos = positions.view(batch_size, 3)
            # else:
            #     target_pos = batch["ee_keyframe_pos"]
            #     pred_ee_pos = perturbed_crop_location + positions.view(batch_size, 3)
            # if self.ori_type == "rpy":
            #     pred_ee_ori = orientations.view(batch_size, 3)
            #     ori_loss = ((pred_ee_ori - target_ee_angles) ** 2).sum()
            # elif self.ori_type == "quaternion":
            # pred_ee_ori = orientations.view(batch_size, 4)
            # target_ee_angles = target_ee_angles.view(batch_size, 4)
            # ori_loss = quaternion_distance(pred_ee_ori, target_ee_angles).sum()

            # pos_loss = ((target_pos - pred_ee_pos) ** 2).sum()

            # classification loss applied to the gripper targets
            # gripper_loss = self.classify_loss(
            # pred_gripper_act.view(-1), target_gripper_state.view(-1)
            # )
            # add up all the losses
            pos_loss /= 3
            ori_loss /= 3
            gripper_loss /= 3
            loss = (
                self.pos_wt * pos_loss
                + self.ori_wt * ori_loss
                + self.gripper_wt * gripper_loss
            )

            tot_pos_loss = tot_pos_loss + pos_loss.item()
            tot_ori_loss = tot_ori_loss + ori_loss.item()
            tot_gripper_loss += gripper_loss.item()

            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            steps += 1

        # breakpoint()
        # print()
        # print("------ Orientation debug info ------")
        # print("Trial was:", batch["trial_name"][0])
        # print("Cmd =", cmd)
        # if self.ori_type == "quaternion":
        #     import trimesh.transformations as tra
        #
        #     T0 = tra.quaternion_matrix(pred_ee_ori[0].detach().cpu().numpy())
        #     T1 = tra.quaternion_matrix(target_ee_angles[0].detach().cpu().numpy())
        #     T1_inv = tra.inverse_matrix(T1)
        #     T01 = T0 @ T1_inv
        #     angles = tra.euler_from_matrix(T01)
        #     print("relative angles =", angles)
        # print("pred ori =", pred_ee_ori[0].detach().cpu().numpy())
        # print("trgt ori =", target_ee_angles[0].detach().cpu().numpy())
        # print()
        # print("pred pos =", pred_ee_pos[0].detach().cpu().numpy())
        # print("trgt pos =", target_pos[0].detach().cpu().numpy())
        return (
            total_loss / steps,
            tot_pos_loss / steps,
            tot_ori_loss / steps,
            tot_gripper_loss / steps,
        )

    def show_validation_on_sensor(self, data, viz=False):
        """
        input is a dict containing raw output from
        """
        self.eval()
        data = self.to_device(data)
        # rgb = data["rgb"]
        # xyz = data["xyz"]
        rgb2 = data["rgb_downsampled"]
        xyz2 = data["xyz_downsampled"]
        cmd = data["cmd"]
        # self.show_pred_and_grnd_truth(
        #     crop_xyz.detach().cpu().numpy(),
        #     crop_rgb.detach().cpu().numpy(),
        #     delta_ee_pos.detach().cpu().numpy().reshape(3, 1),
        #     pred_ori,
        #     query_pt.detach().cpu().numpy().reshape(3, 1),
        #     None,
        #     None,
        #     viewpt=self.cam_view,
        #     save=False,
        # )
        proprio = data["proprio"]
        crop_xyz = data["xyz_crop"]
        crop_rgb = data["rgb_crop"]
        query_pt = data["query_pt"]
        # query_pt = data["crop_ref_ee_keyframe_pos"]
        print("--- ", cmd, " ---")
        (delta_ee_pos, abs_ee_ori, gripper_state,) = self.forward(
            crop_xyz,
            crop_rgb,
            proprio,
            cmd,
        )
        # format pos and ori the right way
        pred_pos = query_pt + delta_ee_pos
        # pred_ori = compute_rotation_matrix_from_ortho6d(
        #     abs_ee_ori.view(-1, 6)
        # ).view(3, 3)
        if self.ori_type == "rpy":
            abs_ee_ori_np = abs_ee_ori.detach().cpu().numpy()
            abs_ee_ori_np[0, 0] += np.pi
            pred_ori = tra.euler_matrix(
                abs_ee_ori_np[0, 0], abs_ee_ori_np[0, 1], abs_ee_ori_np[0, 2]
            )[:3, :3]
        else:
            # w, x, y, z = abs_ee_ori_np
            pred_ori = tra.quaternion_matrix(abs_ee_ori[0].detach().cpu().numpy())[
                :3, :3
            ]
        # abs_ee_ori_np = abs_ee_ori.detach().cpu().numpy()
        # abs_ee_ori_np[0,0] += np.pi
        # pred_ori = tra.euler_matrix(abs_ee_ori_np[0,0], abs_ee_ori_np[0,1], abs_ee_ori_np[0,2])[:3,:3]

        gripper_state = gripper_state > 0.5

        # show point-cloud with coordinate frame where ee should be
        print(f"{cmd}")
        print(f"Predicted gripper state: {gripper_state}")
        self.show_pred_and_grnd_truth(
            xyz2,
            rgb2,
            pred_pos.detach().cpu().numpy().reshape(3, 1),
            pred_ori,
            query_pt.detach().cpu().numpy().reshape(3, 1),
            None,
            None,
            viewpt=self.cam_view,
            save=False,
        )
        self.show_pred_and_grnd_truth(
            crop_xyz.detach().cpu().numpy(),
            crop_rgb.detach().cpu().numpy(),
            delta_ee_pos.detach().cpu().numpy().reshape(3, 1),
            pred_ori,
            query_pt.detach().cpu().numpy().reshape(3, 1),
            None,
            None,
            viewpt=self.cam_view,
            save=False,
        )

        return {
            "predicted_pos": pred_pos.detach().cpu().numpy()[0],
            "predicted_ori": pred_ori,
            "gripper_act": gripper_state.detach().cpu().numpy()[0],
        }

    def show_validation(
        self,
        valid_data,
        viz=False,
        epoch=None,
        save=False,
        debug_regression_training=False,
    ):
        """
        Show some validation visualizations

        debug_regression_training: swap out training data
        """
        self.eval()
        metrics = {"cmd": [], "pos": [], "ori": []}
        for i, batch in enumerate(valid_data):
            batch = self.to_device(batch)
            batch_size = 1
            crop_xyz = batch["xyz_crop"][0]
            crop_rgb = batch["rgb_crop"][0]
            perturbed_crop_location = batch["perturbed_crop_location"]
            num_keypoints = batch["num_keypoints"][0]

            pos_loss = 0
            ori_loss = 0
            gripper_loss = 0
            pos_error = 0
            num_samples = 1

            for t in range(num_keypoints):
                if t == 0:
                    hidden = torch.zeros(1, self.hidden_dim).to(self.device)

                proprio, time_step, cmd = self.get_keypoint(batch, t)
                target_pos, target_ori, target_g = self.get_targets(batch, t)
                print(proprio, time_step, cmd)

                # Run the predictor - get positions and orientations for the model
                if self.use_mdn:
                    (pos_sigma, pos_mu, orientation, gripper, hidden,) = self.forward(
                        crop_xyz, crop_rgb, proprio, time_step, cmd, hidden
                    )
                    target_pos = target_pos.view(num_samples, 3)
                    pos_loss += mdn_loss(pos_sigma, pos_mu, target_pos)
                else:
                    position, orientation, gripper, hidden = self.forward(
                        crop_xyz, crop_rgb, proprio, time_step, cmd, hidden
                    )
                    position = position.view(1, 3)
                    target_pos = target_pos.view(1, 3)
                    pos_loss += ((position - target_pos) ** 2).sum()
                orientation = orientation.view(1, 4)
                target_ori = target_ori.view(1, 4)

                ori_loss += quaternion_distance(orientation, target_ori).sum()
                gripper_loss += self.classify_loss(gripper.view(-1), target_g.view(-1))

                if self.use_mdn:
                    # get positions out of mdn mixture
                    print(pos_sigma, pos_mu)
                    position = sample(pos_sigma, pos_mu, self.device)
                    position = pos_mu.view(3, 1)

                pos_error += np.linalg.norm(
                    target_pos.detach().cpu().numpy()
                    - position.detach().cpu().numpy()
                    - position.detach().cpu().numpy()
                ).sum()
                # create viz variables
                viz_position = (position).detach().cpu().numpy()
                pred_ori_R = tra.quaternion_matrix(orientation.detach().cpu().numpy())[
                    :3, :3
                ]
                viz_target_pos = target_pos.detach().cpu().numpy()
                viz_target_ori = tra.quaternion_matrix(
                    target_ori.detach().cpu().numpy()
                )[:3, :3]
                # show point-cloud with coordinate frame where ee should be
                print(f"{cmd}")
                print(f"Predicted gripper state: {gripper}")
                self.show_pred_and_grnd_truth(
                    crop_xyz.detach().cpu().numpy(),
                    crop_rgb.detach().cpu().numpy(),
                    viz_position.reshape(3, 1),
                    pred_ori_R,
                    perturbed_crop_location.detach().cpu().numpy().reshape(3, 1),
                    viz_target_pos.reshape(3, 1),
                    viz_target_ori.reshape(3, 3),
                    save=save,
                    i=i,
                )
            print("------")
            print(f"pos_error: {pos_error / 3}")
            print(f"pos_loss: {pos_loss / 3}")

            # get input data
        #     xyz = batch["xyz"][0]
        #     rgb = batch["rgb"][0]
        #     xyz_dash = batch["xyz_downsampled"][0]
        #     rgb_dash = batch["rgb_downsampled"][0]
        #     crop_xyz = batch["xyz_crop"][0]
        #     crop_rgb = batch["rgb_crop"][0]
        #     proprio = batch["proprio"][0]
        #     cmd = batch["cmd"]
        #
        #     # extract supervision terms
        #     target_pos = batch["ee_keyframe_pos"]
        #     target_ori = batch["ee_keyframe_ori"]
        #     query_idx = batch["closest_pos_idx"][0]
        #     query_pt = batch["closest_pos"][0]
        #     # angles = tra.euler_from_matrix(crop_ee_keyframe[:3, :3])
        #
        #     print()
        #     print("-" * 8, i, "-" * 8)
        #     print("Trial was:", batch["trial_name"][0])
        #     print("Cmd was:  ", cmd)
        #     print(f"Gripper-state, gripper-width, timestep: {proprio}")
        #
        #     crop_target_pos = batch["ee_keyframe_pos_crop"][0]
        #     crop_target_ori = batch["ee_keyframe_ori_crop"][0]
        #     target_angles = batch["target_ee_angles"]
        #     crop_location = batch["perturbed_crop_location"]
        #
        #     (delta_ee_pos, abs_ee_ori, gripper_state,) = self.forward(
        #         crop_xyz,
        #         crop_rgb,
        #         proprio,
        #         cmd,
        #     )
        #
        #     # format pos and ori the right way
        #     pred_pos = delta_ee_pos
        #     # Create the orientation and convert it from whatever its native form is
        #     if self.ori_type == "rpy":
        #         # Roll pitch yaw setup - might need to skip
        #         abs_ee_ori_np = abs_ee_ori.detach().cpu().numpy()
        #         abs_ee_ori_np[0, 0] += np.pi
        #         pred_ori = tra.euler_matrix(
        #             abs_ee_ori_np[0, 0], abs_ee_ori_np[0, 1], abs_ee_ori_np[0, 2]
        #         )
        #         if debug_regression_training:
        #             raise NotImplementedError()
        #         raise NotImplementedError("we only support quaternions right now")
        #     else:
        #         # Convert the quaternion setup into a pose matrix that we can use
        #         # w, x, y, z = abs_ee_ori_np
        #         pred_ori = tra.quaternion_matrix(abs_ee_ori[0].detach().cpu().numpy())
        #
        #     if self.multi_head:
        #         iterations = 3
        #         pred_pos = pred_pos[0]
        #         target_angles = batch["target_ee_angles"][0]
        #     else:
        #         iterations = 1
        #
        #     i = 0
        #     while i < iterations:
        #         # Create copies for debugging and visualization
        #         if self.multi_head:
        #             pred_ori_R = pred_ori[i, :3, :3]
        #             pred_ori_4x4 = np.copy(pred_ori[i])
        #             viz_target_pos = crop_target_pos[i]
        #             viz_target_ori = crop_target_ori[i]
        #         else:
        #             pred_ori_R = pred_ori[:3, :3]
        #             pred_ori_4x4 = np.copy(pred_ori)
        #             viz_target_pos = target_pos - crop_location
        #             viz_target_ori = crop_target_ori
        #         T1 = tra.quaternion_matrix(target_angles[i].detach().cpu().numpy())
        #         T1_inv = tra.inverse_matrix(T1)
        #         T01 = pred_ori_4x4 @ T1_inv
        #         angles = tra.euler_from_matrix(T01)
        #         ori_error = np.sum(angles) / 3
        #         print("Error in relative angles = ", angles)
        #         if self.multi_head:
        #             pos_error = np.linalg.norm(
        #                 crop_target_pos[i].detach().cpu().numpy()
        #                 - pred_pos[i].detach().cpu().numpy()
        #             )
        #         else:
        #             pos_error = np.linalg.norm(
        #                 viz_target_pos.detach().cpu().numpy()
        #                 - pred_pos[i].detach().cpu().numpy()
        #             )
        #         print(f"Error in meters: {pos_error}")
        #
        #         gripper_state = gripper_state > 0.5
        #
        #         # show point-cloud with coordinate frame where ee should be
        #         print(f"{cmd}")
        #         print(f"Predicted gripper state: {gripper_state}")
        #         self.show_pred_and_grnd_truth(
        #             crop_xyz.detach().cpu().numpy(),
        #             crop_rgb.detach().cpu().numpy(),
        #             pred_pos[i].detach().cpu().numpy().reshape(3, 1),
        #             pred_ori_R,
        #             query_pt.detach().cpu().numpy().reshape(3, 1),
        #             viz_target_pos.detach().cpu().numpy().reshape(3, 1),
        #             viz_target_ori.detach().cpu().numpy().reshape(3, 3),
        #             save=save,
        #             i=i,
        #             epoch=epoch,
        #         )
        #         i += 1
        #
        #     metrics["cmd"].append(cmd)
        #     metrics["pos"].append(float(pos_error))
        #     metrics["ori"].append(float(ori_error))
        # pprint(metrics)
        # todaydate = datetime.date.today()
        # time = datetime.datetime.now().strftime("%H_%M")
        # output_dir = f"./outputs/{todaydate}/{self.name}/"
        # output_file = f"output_{time}.json"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # with open(os.path.join(output_dir, output_file), "w") as f:
        #     json.dump(metrics, f, indent=4)


@hydra.main(
    version_base=None, config_path="./conf", config_name="action_predictor_training"
)
def main(cfg):
    # Speed up training by configuring the number of workers
    num_workers = 8 if not cfg.debug else 0
    B = 1

    # create model, load weights for classifier
    model = ActionPredictionModule(cfg)
    model.to(model.device)
    optimizer = model.get_optimizer()
    scheduler = ReduceLROnPlateau(optimizer, patience=3)

    if cfg.source in ["franka", "stretch"]:
        # Update splits - only used here
        if cfg.split:
            with open(cfg.split, "r") as f:
                train_test_split = yaml.safe_load(f)
            print(train_test_split)
            train_list = train_test_split["train"]
            valid_list = train_test_split["val"]
            test_list = train_test_split["test"]
        else:
            print(
                "No split file specified, loading everything or based on hardcoded trial_list"
            )
            train_list = None
            valid_list = None
            test_list = None

        # Create datasets
        # train_dir = robopen_data_dir
        # valid_dir = robopen_data_dir
        # Dataset = RobotDataset
        train_dataset = RobotDataset(
            cfg.data_dir,
            num_pts=cfg.num_pts,
            data_augmentation=cfg.data_augmentation,  # (not validate),
            ori_dr_range=np.pi / 8,
            # first_frame_as_input=True,
            keypoint_range=[cfg.action_idx] if cfg.action_idx > -1 else [0, 1, 2],
            trial_list=train_list,
            orientation_type=cfg.orientation_type,
            multi_step=cfg.multi_step,
            template=cfg.template,
            autoregressive=True,
            time_as_one_hot=True,
            per_action_cmd=cfg.per_action_cmd,
        )
        valid_dataset = RobotDataset(
            cfg.data_dir,
            num_pts=cfg.num_pts,
            data_augmentation=False,
            # first_frame_as_input=True,
            trial_list=valid_list,
            keypoint_range=[cfg.action_idx] if cfg.action_idx > -1 else [0, 1, 2],
            orientation_type=cfg.orientation_type,
            multi_step=cfg.multi_step,
            template=cfg.template,
            autoregressive=True,
            time_as_one_hot=True,
            per_action_cmd=cfg.per_action_cmd,
        )
        test_dataset = RobotDataset(
            cfg.data_dir,
            num_pts=cfg.num_pts,
            data_augmentation=False,
            # first_frame_as_input=True,
            keypoint_range=[cfg.action_idx] if cfg.action_idx > -1 else [0, 1, 2],
            trial_list=test_list,
            orientation_type=cfg.orientation_type,
            multi_step=cfg.multi_step,
            template=cfg.template,
            autoregressive=True,
            time_as_one_hot=True,
            per_action_cmd=cfg.per_action_cmd,
        )
    else:
        train_dir = train_dataset_dir
        valid_dir = valid_dataset_dir
        Dataset = RLBenchDataset

        # load data
        train_dataset = Dataset(
            train_dir,
            num_pts=cfg.num_pts,
            data_augmentation=(not cfg.validate),
            ori_dr_range=np.pi / 8,
            verbose=True,
            # first_keypoint_only=(first_keypoint_only or multi_head),
            orientation_type=cfg.orientation_type,
            multi_step=cfg.multi_step,
        )
        valid_dataset = Dataset(
            valid_dir,
            data_augmentation=False,
            num_pts=cfg.num_pts,
            verbose=True,
            # first_keypoint_only=(first_keypoint_only or multi_head),
            orientation_type=cfg.orientation_type,
            multi_step=cfg.multi_step,
        )
        test_dataset = valid_dataset

    # Create data loaders
    train_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=B, num_workers=num_workers, shuffle=True
    )
    valid_data = torch.utils.data.DataLoader(
        valid_dataset, batch_size=B, num_workers=num_workers, shuffle=False
    )
    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=B, num_workers=num_workers, shuffle=False
    )

    if cfg.validate:
        # we need to predict for validation data and show point-cloud
        # with regressed position and orientation for the ee
        if not cfg.load:
            model.load_weights(model.get_best_name())
        else:
            model.load_weights(cfg.load)
        model.show_validation(valid_data, viz=True)
    else:
        if cfg.wandb:
            date_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
            wandb.init(project="action_predictor", name=f"{cfg.name}_{date_time}")
            wandb.config.query_radius = model._query_radius
            # wandb.config.voxelization_scheme = [
            #     test_dataset._voxel_size,
            #     test_dataset._voxel_size_2,
            # ]
            wandb.config.pos_wt = model.pos_wt
            wandb.config.ori_wt = model.ori_wt
            wandb.config.gripper_wt = model.gripper_wt
        best_valid_loss = float("Inf")
        model.start_time = time()
        for epoch in range(1, cfg.max_iter + 1):
            # model.curr_epoch = epoch
            (
                tot_loss,
                pos_train_loss,
                ori_train_loss,
                g_train_loss,
            ) = model.do_epoch(train_data, optimizer, train=True)
            print("total loss =", tot_loss)
            print("  pos loss =", pos_train_loss)
            print("  ori loss =", ori_train_loss)
            print("grasp loss =", g_train_loss)
            train_loss = tot_loss
            with torch.no_grad():
                (
                    valid_loss,
                    pos_loss,
                    ori_loss,
                    g_valid_loss,
                ) = model.do_epoch(valid_data, optimizer, train=False)
            print("-----")
            print("valid_pos_loss:", pos_loss)
            print("valid_ori_loss:", ori_loss)
            print("valid_grp_loss::", g_valid_loss)
            if cfg.wandb:
                wandb.log(
                    {
                        "train": train_loss,
                        "valid": valid_loss,
                        "train_pos_err": pos_train_loss,
                        "train_ori_loss": ori_train_loss,
                        "valid_pos_err": pos_loss,
                        "valid_ori_loss": ori_loss,
                        "g_train": g_train_loss,
                        "g_valid": g_valid_loss,
                    }
                )

            print(
                f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}"
            )
            scheduler.step(valid_loss)
            best_valid_loss, updated = model.smart_save(
                epoch, valid_loss, best_valid_loss
            )
            reload_model = True
            if reload_model and not updated:
                print("--> reload state dict from:", model.get_best_name())
                print(f"--> best loss was {best_valid_loss}")
                model.load_state_dict(torch.load(model.get_best_name()))
            # if run_for and (time() - start_time) > run_for:
            #     print(f" --> Stopping training after {run_for} seconds")
            #     break


if __name__ == "__main__":
    main()
