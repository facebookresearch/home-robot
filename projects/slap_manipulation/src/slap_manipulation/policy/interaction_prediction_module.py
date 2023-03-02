import argparse
import datetime
import os
import random
from pprint import pprint
from time import time

import clip
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import wandb
import yaml
from home_robot.utils.point_cloud import numpy_to_pcd, show_pcd, show_point_cloud
from perceiver_pytorch.perceiver_io import (
    FeedForward,
    PreNorm,
    cache_fn,
    default,
    exists,
)
from torch_geometric.nn import MLP, Linear, PointConv, radius
from tqdm import tqdm

from slap_manipulation.dataloaders.rlbench_loader import RLBenchDataset
from slap_manipulation.dataloaders.robot_loader import (
    RobotDataset,
    show_point_cloud_with_keypt_and_closest_pt,
)

# Optimizer from ARM
from slap_manipulation.optim.lamb import Lamb

# Network blocks from ARM
# Perceiver networks
from slap_manipulation.policy.components import (
    Attention,
    DenseBlock,
    PositionalEncoding,
    SAModule,
)

np.random.seed(0)
torch.manual_seed(0)

random.seed(0)

# NUM_LATENTS = 512
NUM_LATENTS = 256
# NUM_LATENTS = 128
LATENT_DIM = 512
# LATENT_DIM = 256


class LocalityLoss(torch.nn.Module):
    """Compute center point as weighted mean of all points
    penalize anything farther away from this
    make sure that we're choosing the right locations for action"""

    def __init__(self, wt=1.0, spread=0.5):
        super().__init__()
        self.wt = wt
        self.spread = spread

    def forward(self, xyz, scores):
        """penalize dispersal - find center, penalize distance from that
        scaled according to the spread provided (high self.spread = lower penality)"""
        B, C = xyz.shape
        scores = torch.softmax(scores, dim=-1)
        xyz_scores = scores.view(B, 1).repeat(1, 3)
        center = (xyz * xyz_scores).sum(dim=0).view(1, C).repeat(B, 1)
        dists = ((xyz - center) ** 2).sum(dim=-1)
        loss = self.wt * (dists / self.spread).mean()
        return loss


class SupervisedLocalityLoss(torch.nn.Module):
    """Compute center point as weighted mean of all points
    penalize anything farther away from this
    make sure that we're choosing the right locations for action"""

    def __init__(self, wt=1.0, spread=0.5):
        super().__init__()
        self.wt = wt
        self.spread = spread

    def forward(self, goal_pt, xyz, scores):
        """penalize dispersal - find center, penalize distance from that
        scaled according to the spread provided (high self.spread = lower penality)"""
        B, C = xyz.shape
        scores = torch.softmax(scores, dim=-1)
        # scores = torch.sigmoid(scores)
        center = goal_pt.view(1, C).repeat(B, 1)
        dists = ((xyz - center) ** 2).sum(dim=-1)
        dists = dists * scores
        loss = self.wt * (dists / self.spread).mean()
        return loss


class IPModule(torch.nn.Module):
    """
    Query evaluator.
    This is modified based on our previous version of the code.
    We downsample the input point-cloud and use classification schema
    to choose the point in that downsampled version closest to next
    keypoint. Supervision is done based on ground truth of the closest point

    old description:
    This is modified based on our previous version of the code.
    The idea is that we have a normal segmentation network, instead of predicting
    offsets and masks, we just pick a volume from a set which we expect to be most
    relevant and then predict a continuous offset from that.
    """

    # for visualizations
    cam_view = {
        "front": [-0.89795424592554529, 0.047678244807235863, 0.43749852250766141],
        "lookat": [0.33531651482385966, 0.048464899929339826, 0.54704503365806367],
        "up": [0.43890929711345494, 0.024286597087151203, 0.89820308956788786],
        "zoom": 0.43999999999999972,
    }

    def __init__(
        self,
        optimizer_type: str = "lamb",
        lr: float = 0.002,
        lambda_weight_l2: float = 0.000001,
        activation="lrelu",
        num_latents=NUM_LATENTS,
        latent_dim=LATENT_DIM,
        transformer_iterations=1,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        transformer_depth=6,  # Number of transformer layers?
        xent_loss=True,
        use_proprio=True,
        use_final_sa=False,
        learned_pos_encoding=False,
        name="v1_classification",
        skip_ambiguous_pts=False,
        locality_loss_wt=1e-0,
        locality_loss_spread=0.25,
        dry_run=False,
    ):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._skip_ambiguous_pts = skip_ambiguous_pts
        self._optimizer_type = optimizer_type
        self._lr = lr
        self._lambda_weight_l2 = lambda_weight_l2
        self.iterations = transformer_iterations
        self.latent_dim = latent_dim
        self.xent_loss = xent_loss
        if self.xent_loss:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = torch.nn.BCELoss()

        # self.locality_loss_fn = LocalityLoss(wt=locality_loss_wt,
        #                                     spread=locality_loss_spread)
        self.locality_loss_fn = SupervisedLocalityLoss(
            wt=locality_loss_wt, spread=locality_loss_spread
        )

        self._query_radius = 0.2
        self._max_neighbor = 64  # default from pointnet2 code
        # for mixing proprio and images
        self.proprio_dim_size = 3
        self.im_channels = 64
        self.use_proprio = use_proprio

        # setup dimensionality of channels
        self.pc_dim = 64
        if self.use_proprio:
            self.tf_dim = 2 * self.pc_dim
        else:
            self.tf_dim = self.pc_dim

        # for classification component
        # self.sa1_module = SAModule(0.05, 128, self.device, MLP([6, 64, 64, 128]))
        self.sa1_module = SAModule(
            0.5 * self._query_radius,
            128,
            self.device,
            MLP([6, 64, 128], batch_norm=False),
        )
        self.sa2_module = SAModule(
            # 0.1, 64, self.device, MLP([128 + 3, 128, 256, 512])
            self._query_radius,
            64,
            self.device,
            # MLP([128 + 3, 256, 512, self.pc_dim], batch_norm=False),
            MLP([128 + 3, 256, 512], batch_norm=False),
            # MLP([128 + 3, 128, 128]),
        )
        # After the aggregation in SA module, then we can project down to whatever feature set
        # we actually want to use in the transformer model.
        self.pt_enc = nn.Sequential(nn.Linear(512, self.pc_dim), nn.LeakyReLU())

        # Do something with this
        self.use_final_sa = use_final_sa
        if self.use_final_sa:
            self.final_mlp = nn.Sequential(
                nn.Linear(2 * self.tf_dim, 2 * self.tf_dim),
                nn.LeakyReLU(),
                nn.Linear(2 * self.tf_dim, self.tf_dim),
            )
            self.to_activation = torch.nn.Linear(128, 1, bias=False)
        else:
            self.to_activation = torch.nn.Linear(self.tf_dim, 1)
        self.softmax = torch.nn.Softmax(dim=0)
        with torch.no_grad():
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )

        # learnable positional encoding
        # Unlike eg in peract, this ONLY applies to the language
        lang_emb_dim, lang_max_seq_len = 512, 77
        # Positional encoding is only applied to the language term(s)
        self.learned_pos_encoding = learned_pos_encoding
        if self.learned_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, lang_max_seq_len, self.tf_dim)
            )
        else:
            self.pos_encoding = PositionalEncoding(self.tf_dim)

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            self.proprio_dim_size,
            self.pc_dim,
            norm="layer",
            activation=activation,
        )

        # lang preprocess
        self.lang_preprocess = torch.nn.Sequential(
            # nn.Linear(lang_emb_dim, lang_emb_dim),
            # nn.LeakyReLU(),
            nn.Linear(lang_emb_dim, self.tf_dim),
        )

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim))

        # Inputs from our preprocessing step
        self.input_dim_before_seq = self.tf_dim

        # Create the transformer
        # encoder cross attention
        # We will cross attend back
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    self.latent_dim,
                    Attention(
                        self.latent_dim,
                        self.input_dim_before_seq,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        dropout=input_dropout,
                    ),
                    context_dim=self.input_dim_before_seq,
                ),
                PreNorm(self.latent_dim, FeedForward(self.latent_dim)),
            ]
        )

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        # Create some lambdas for filling out latent attentions
        # This of course is from PerAct
        def get_latent_attn():
            return PreNorm(
                self.latent_dim,
                Attention(
                    self.latent_dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=attn_dropout,
                ),
            )

        def get_latent_ff():
            return PreNorm(latent_dim, FeedForward(latent_dim))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        for i in range(transformer_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(
            self.input_dim_before_seq,
            Attention(
                self.input_dim_before_seq,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=decoder_dropout,
            ),
            context_dim=latent_dim,
        )
        if not dry_run:
            self.setup_training()
            self.start_time = 0.0

    def setup_training(self):
        # get today's date
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = self.name + "_" + date_time
        # append folder name to current working dir
        path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
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

    def to_device(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                new_batch[k] = v
            else:
                new_batch[k] = v.to(self.device)
        return new_batch

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

    def _preprocess_input(self, xyz, rgb):
        pcd = numpy_to_pcd(xyz.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        downpcd = pcd.voxel_down_sample(voxel_size=self._voxel_size)
        down_xyz = torch.Tensor(np.asarray(downpcd.points)).to(self.device)
        down_rgb = torch.Tensor(np.asarray(downpcd.colors)).to(self.device)
        return down_xyz, down_rgb

    def eval(self, data):
        return None

    def predict(self, feat, xyz, lang):
        """
        helper function for predicting index of closest voxel and encoded spatial features

        feat: tuple (rgb, proprio)
        xyz: point-cloud locations for each feat
        lang: language annotation which is encoded
        """
        rgb = feat[0]
        proprio = feat[1]
        down_xyz, down_rgb = self._preprocess_input(xyz, rgb)
        classification_probs, xyz, rgb = self.forward(
            rgb,
            down_rgb,
            xyz,
            down_xyz,
            lang,
            proprio,
        )
        return (self.predict_closest_idx(classification_probs)[0], xyz, rgb)

    def predict_closest_idx(self, classification_probs) -> torch.Tensor:
        # predict the closest centroid
        predicted_idx = torch.argmax(classification_probs, dim=-1)
        return predicted_idx

    def show_prediction_with_grnd_truth(
        self,
        xyz,
        rgb,
        pred_pos,
        grnd_truth_pos=None,
        show_input=False,
        i=-1,
        save=False,
        viewpt={},
    ):
        if torch.any(rgb) > 1:
            rgb = rgb / 255.0
        pcd = numpy_to_pcd(xyz.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        geoms = [pcd]
        closest_pt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        closest_pt_sphere.translate(pred_pos)
        closest_pt_sphere.paint_uniform_color([1, 0.706, 0])
        geoms.append(closest_pt_sphere)
        if grnd_truth_pos is not None:
            grnd_truth = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            grnd_truth.translate(grnd_truth_pos.reshape(3, 1))
            grnd_truth.paint_uniform_color([1, 0, 1])
            geoms.append(grnd_truth)
        else:
            print("No ground truth available")
        o3d.visualization.draw_geometries(geoms)
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # for geom in geoms:
        #     vis.add_geometry(geom)
        #     vis.update_geometry(geom)
        # if viewpt:
        #     ctr = vis.get_view_control()
        #     ctr.set_front(self.cam_view["front"])
        #     ctr.set_lookat(self.cam_view["lookat"])
        #     ctr.set_up(self.cam_view["up"])
        #     ctr.set_zoom(self.cam_view["zoom"])
        # if save:
        #     vis.poll_events()
        #     vis.update_renderer()
        #     vis.capture_screen_image(f"/home/robopen08/.larp/{self.name}_{i}.png")
        #     vis.destroy_window()
        # else:
        #     vis.run()
        #     vis.destroy_window()
        # del vis
        # if viewpt:
        #     del ctr

    def show_validation_on_sensor(self, data, viz=False):
        """
        input is a dict containing raw output from
        """
        print("In validation loop")
        self.eval()
        data = self.to_device(data)
        rgb = data["rgb"]
        xyz = data["xyz"]
        rgb2 = data["rgb_downsampled"]
        xyz2 = data["xyz_downsampled"]
        cmd = data["cmd"]
        proprio = data["proprio"]
        print("--- ", cmd, " ---")
        with torch.no_grad():
            classification_probs, down_xyz, down_rgb = self.forward(
                rgb,
                rgb2,
                xyz,
                xyz2,
                cmd,
                proprio,
            )
        if viz:
            show_point_cloud(xyz2, rgb2, viewpt=self.cam_view)
            _, mask = torch.sort(classification_probs, descending=True)
            top_10 = int(classification_probs.shape[1] * 0.05)
            mask = mask.squeeze()[:top_10].detach().cpu().numpy()
            # mask = (classification_probs > thresh).detach().cpu().numpy()
            new_rgb = rgb2.detach().cpu().numpy().copy()
            new_rgb[mask] = np.array([1, 0, 0]).reshape(1, 3)
            show_point_cloud(xyz2, new_rgb, viewpt=self.cam_view)

        predicted_idx = torch.argmax(classification_probs)
        predict_pos = (
            xyz2[predicted_idx.detach().cpu().numpy()]
            .detach()
            .cpu()
            .numpy()
            .reshape(3, 1)
        )
        print("found predicted-position")

        # also show the target closest point
        self.show_prediction_with_grnd_truth(
            xyz, rgb, predict_pos, None, save=False, viewpt=self.cam_view
        )
        # predicted_idx = torch.argmax(classification_probs, dim=-1)
        # predict_pos = xyz2[predicted_idx.detach().cpu().numpy()]
        # # also show the target closest point
        # self.show_prediction(
        #     xyz2,
        #     rgb2,
        #     xyz2[predicted_idx[0]].detach().cpu().numpy(),
        # )
        return predict_pos

    def show_validation(self, loader, viz=False, save=False, viz_mask=False):
        self.eval()
        metrics = {"pos": [], "cmd": []}
        for i, batch in enumerate(tqdm(loader, ncols=50)):
            batch = self.to_device(batch)
            rgb = batch["rgb"][0]
            xyz = batch["xyz"][0]
            rgb2 = batch["rgb_downsampled"][0]
            xyz2 = batch["xyz_downsampled"][0]
            cmd = batch["cmd"]
            proprio = batch["proprio"][0]
            target_pos = batch["closest_voxel"][0]
            metrics["cmd"].append(cmd)

            print()
            print("---", i, "---")
            print(cmd)

            # down_xyz, down_rgb = self._preprocess_input(xyz, rgb)
            classification_probs, down_xyz, down_rgb = self.forward(
                rgb,
                rgb2,
                xyz,
                xyz2,
                cmd,
                proprio,
            )

            if viz_mask:
                _, mask = torch.sort(classification_probs, descending=True)
                ten_percent = int(int(classification_probs.shape[1]) * 0.05)
                mask = mask[0, :ten_percent].detach().cpu().numpy()
                new_rgb = rgb2.detach().cpu().numpy().copy()
                new_rgb[mask.reshape(-1)] = np.array([1, 0, 0]).reshape(1, 3)
                show_point_cloud(xyz2, rgb2, viewpt=self.cam_view)
                show_point_cloud(xyz2, new_rgb, viewpt=self.cam_view)

            predicted_idx = self.predict_closest_idx(classification_probs)
            predict_pos = xyz2[predicted_idx.detach().cpu().numpy()]
            print(f"Target position: {target_pos}")
            print(f"Predicted position: {predict_pos}")
            error = np.linalg.norm(
                target_pos.detach().cpu().numpy() - predict_pos.detach().cpu().numpy()
            )
            print(f"Error in meters: {error}")
            metrics["pos"].append(float(error))

            # also show the target closest point
            if viz:
                self.show_prediction_with_grnd_truth(
                    xyz,
                    rgb,
                    xyz2[predicted_idx[0]].detach().cpu().numpy(),
                    target_pos.detach().cpu().numpy(),
                    i=i,
                    save=save,
                )
        pprint(metrics)
        todaydate = datetime.date.today()
        time = datetime.datetime.now().strftime("%H_%M")
        output_dir = f"./outputs/{todaydate}/{self.name}/"
        output_file = f"output_{time}.json"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, output_file), "w") as f:
            json.dump(metrics, f, indent=4)

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
        lang = lang.view(batch_size * lang_seq_len, -1)
        if self.learned_pos_encoding:
            lang = self.lang_preprocess(lang) + self.pos_encoding
        else:
            lang = self.lang_preprocess(lang)
            lang = self.pos_encoding(lang)
        lang = lang.view(batch_size, lang_seq_len, -1)

        return lang

    def do_epoch(self, loader, optimizer=None, train=True):
        if train:
            self.train()
            if optimizer is None:
                raise RuntimeError("hard to train without an optimizer")
        else:
            self.eval()

        total_loss = 0.0
        total_dist = 0.0
        total_steps = 0

        for _, batch in enumerate(tqdm(loader, ncols=50)):
            # get the inputs
            if not batch["data_ok_status"]:
                continue
            batch = self.to_device(batch)
            optimizer.zero_grad()
            rgb = batch["rgb"][0]  # N x 3
            xyz = batch["xyz"][0]  # N x 3
            down_xyz = batch["xyz_downsampled"][0]
            down_rgb = batch["rgb_downsampled"][0]
            lang = batch["cmd"]  # list of 1
            if self.use_proprio:
                proprio = batch["proprio"][0]
            else:
                proprio = None

            # extract supervision terms
            target_idx = batch["closest_voxel_idx"][0]
            target_pos = batch["closest_voxel"]
            ee_keyframe = batch["ee_keyframe_pos"][0]
            ee_keyframe_rot = batch["ee_keyframe_ori"][0]

            # Visualize
            # show_point_cloud(
            #        down_xyz.detach().cpu().numpy(),
            #        down_rgb.detach().cpu().numpy(),
            #        orig=np.zeros(3))
            if False:
                show_point_cloud_with_keypt_and_closest_pt(
                    down_xyz.detach().cpu().numpy(),
                    down_rgb.detach().cpu().numpy(),
                    ee_keyframe.detach().cpu().numpy().reshape(3, 1),
                    ee_keyframe_rot.detach().cpu().numpy().reshape(3, 3),
                    target_pos.detach().cpu().numpy().reshape(3, 1),
                )

            # predict the closest centroid
            classification_probs, _, _ = self.forward(
                rgb,
                down_rgb,
                xyz,
                down_xyz,
                lang,
                proprio,
            )
            predicted_idx = torch.argmax(classification_probs, dim=-1)
            dist = np.linalg.norm(
                down_xyz[target_idx].detach().cpu().numpy()
                - down_xyz[predicted_idx].detach().cpu().numpy()
            )
            total_dist += dist

            # compute teh locality loss here
            # Should apply softmax here
            locality_loss = self.locality_loss_fn(
                down_xyz[target_idx], down_xyz, classification_probs
            )

            # backprop loss
            if self.xent_loss:
                target = target_idx
            else:
                target = torch.nn.functional.one_hot(
                    target_idx, num_classes=classification_probs.shape[-1]
                ).float()
                classification_probs = torch.sigmoid(classification_probs)

            # Do some masking
            # Basically - we want to remove probs and targets that arent what we expect here
            if self._skip_ambiguous_pts:
                voxel_mask = batch["xyz_mask"][0].bool().cpu().numpy()
                classification_probs = classification_probs[:, voxel_mask]
                if self.xent_loss:
                    target_idx = target[0].item()
                    new_target_idx = voxel_mask[:target_idx].sum()
                    target = torch.LongTensor([new_target_idx]).to(self.device)
                else:
                    target = target[:, classification_probs]

            loss = self.loss_fn(classification_probs, target) + locality_loss
            # print(float(loss), float(locality_loss))

            # TODO: remove this unused code
            # classification_probs = classification_probs / classification_probs.sum()
            # probs = classification_probs[0][:, None].repeat(1, 3)
            # weighted_xyz = (probs * down_xyz).mean(dim=0)
            # loss += ((weighted_xyz - target_pos) ** 2).sum()

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        print()
        print("---", "train" if train else "valid", "example ---")
        print("skill was:", batch["cmd"])
        print(f"Target idx: {down_xyz[target_idx]}")
        print(f"Predicted idx: {down_xyz[predicted_idx]}")
        print(f"Distance b/w: {dist}")
        print("       loss =", float(loss))
        print("AVG LOSS:", total_loss / total_steps)
        print()

        return total_loss / total_steps, total_dist / total_steps

    def forward(
        self,
        feat,
        sampled_feat,
        pos,
        sampled_pos,
        lang_cmd,
        proprio=None,
        mask=None,
        B=None,
    ):
        """given centroids and pcd, choose the centroid closest to next keypoint"""
        batch_size = B

        # TODO: apply some good normalization here
        # normalize rgb if not already normalized
        if torch.max(feat) > 1 or torch.max(sampled_feat) > 1:
            raise RuntimeError("invalid values for rgb pt cloud features")
            feat = self.normalize_rgb(feat)
        # if torch.max(sampled_feat) > 1:
        #    sampled_feat = self.normalize_rgb(sampled_feat)

        # Extract language. This should let us create more interesting things...
        lang = self.clip_encode_text(lang_cmd)
        batch_size, lang_seq_len, _ = lang.shape

        # get centroids from downsampled point-cloud
        # print(f"Downsampled pcd has {sampled_pos.shape} points")
        feat, pos, batch = self.sa1_module(feat, sampled_feat, pos, sampled_pos, lang)
        # Again process features for sampled positions, based on previous round
        ins, pos, batch = self.sa2_module(feat, sampled_feat, pos, sampled_pos, lang)
        ins = self.pt_enc(ins)

        # combine ins with proprio information
        if self.use_proprio:
            proprio = self.proprio_preprocess(proprio)
            ins = torch.cat(
                (ins, proprio.view(1, self.pc_dim).repeat(ins.shape[0], 1)),
                dim=1,
            )

        # x = self.latents
        # TODO: this needs to add batch dimension and repeat
        x = self.latents[None]
        # TODO: this needs to add batch dimension and repeat
        in_pts = ins
        ins = ins[None]

        # combine ins with language
        num_query_pts = ins.shape[1]
        ins = torch.cat([lang, ins], dim=1)

        cross_attn, cross_ff = self.cross_attend_blocks
        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        # Attend back to the input vector one more time
        latents = self.decoder_cross_attn(ins, context=x)
        # This would grab only the latents NOT corresponding to lang dims
        # So that we can attend to the point in space
        latents = latents[:, lang_seq_len:]

        # Now we COULD upsample back to the original PC
        # Or we could do something else...
        B, num_pts, _ = latents.shape
        latents = latents.view(B * num_pts, -1)

        # Cross as if we are doing this
        if self.use_final_sa:
            feats = torch.cat([latents, in_pts], dim=-1)
            feats = self.final_mlp(feats)
        else:
            feats = latents

        # Linear layer - convert to feature activations
        acts = self.to_activation(feats)
        acts = acts.view(B, num_pts)

        return acts, pos, feats


def parse_args():
    parser = argparse.ArgumentParser("train_robopen_model")
    parser.add_argument("-v", "--validate", action="store_true")
    parser.add_argument("-d", "--datadir", default="./data/small_ds")
    # parser.add_argument('--load', type=str, help='weights to load', default='best_ptt_robopen.pth')
    parser.add_argument("--load", type=str, help="weights to load", default="")
    parser.add_argument("--resume", action="store_true", help="overrides load")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("-t", "--task_name", default="pick_box", type=str)
    parser.add_argument("--loss_fn", choices=["bce", "xent"], default="xent")
    parser.add_argument(
        "-s", "--source", choices=["robopen", "rlbench"], default="rlbench"
    )
    parser.add_argument("--run_for", type=int, help="how long to run for in seconds")
    parser.add_argument(
        "--wandb", action="store_true", help="enable wandb for this run"
    )
    parser.add_argument("--split", help="path to test_val_split", default=None)
    parser.add_argument(
        "-D",
        "--data-augmentation",
        action="store_true",
        help="If the training ds should be augmented with DR",
    )
    parser.add_argument(
        "--color-jitter", help="use color jitter when training", action="store_true"
    )
    # parser.add_argument("--reload", action="store_true", help="reload best val")
    parser.add_argument("--template", default="*.h5")
    args = parser.parse_args()
    return args


# @hydra.main()
def main():
    args = parse_args()
    if args.split is not None:
        with open(args.split, "r") as f:
            train_test_split = yaml.safe_load(f)
        print(train_test_split)
        valid_list = train_test_split["val"]
        test_list = train_test_split["test"]
        train_list = train_test_split["train"]
    else:
        train_test_split = None
        valid_list, test_list = None, None
    wandb_flag = True if args.wandb else False
    # Set up data augmentation
    # This is a warning for you - if things are not going well
    if args.data_augmentation:
        print("-> Using data augmentation on training data.")
    else:
        print("-> NOT using data augmentation.")
    # Set up data loaders
    if args.source == "robopen":
        # Get the robopebn dataset
        task_name = args.task_name
        Dataset = RoboPenDataset
        train_dataset = RoboPenDataset(
            args.datadir,
            trial_list=train_list,
            data_augmentation=args.data_augmentation,
            ori_dr_range=np.pi / 8,
            num_pts=8000,
            random_idx=False,
            keypoint_range=[0, 1, 2],
            color_jitter=args.color_jitter,
            template=args.template,
            # template="**/*.h5",
            dr_factor=5,
        )
        valid_dataset = RoboPenDataset(
            args.datadir,
            num_pts=8000,
            data_augmentation=False,
            trial_list=valid_list,
            keypoint_range=[0, 1, 2],
            color_jitter=False,
            template=args.template,
            # template="**/*.h5",
        )
        test_dataset = RoboPenDataset(
            args.datadir,
            num_pts=8000,
            data_augmentation=False,
            trial_list=valid_list,
            keypoint_range=[0, 1, 2],
            color_jitter=False,
            template=args.template,
            # template="**/*.h5",
        )
    else:
        # get rlbench dataset
        first_keypoint_only = True
        Dataset = RLBenchDataset
        train_dataset = Dataset(
            train_dataset_dir,
            # trial_list=overfit_list,
            num_pts=8000,
            data_augmentation=args.data_augmentation,
            ori_dr_range=np.pi / 8,
            random_idx=False,
            first_keypoint_only=first_keypoint_only,
            color_jitter=args.color_jitter,
        )
        test_dataset = Dataset(
            valid_dataset_dir,
            # trial_list=overfit_list,
            num_pts=8000,
            data_augmentation=False,  # no randomization only real data
            # no random indexes only the input frames (beginning of acts)
            random_idx=False,
            first_keypoint_only=first_keypoint_only,
            color_jitter=False,
        )
        valid_dataset = test_dataset

    # Create data loaders
    num_workers = 8 if not args.debug else 0
    B = 1
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    valid_data = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )
    # load the model
    model = IPModule(
        xent_loss=args.loss_fn == "xent",
        use_proprio=True,
        name=f"classify-{args.task_name}",
    )
    model.to(model.device)

    optimizer = model.get_optimizer()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if args.resume:
        print(f"Resuming by loading the best model: {model.get_best_name()}")
        args.load = model.get_best_name()
    if len(args.load) > 0:
        # load the model now
        model.load_state_dict(torch.load(args.load))
        print("--> loaded last best <--")
    if args.validate:
        # Make sure we load something
        if len(args.load) == 0:
            args.load = "best_%s.pth" % model.name
            print(
                f" --> No model name provided to validate. Using default...{args.load}"
            )

        if len(args.load) > 0:
            # load the model now
            model.load_state_dict(torch.load(args.load))

        with torch.no_grad():
            model.show_validation(train_data, viz=True, viz_mask=True)
    else:
        best_valid_loss = float("Inf")
        print("Starting training")
        if wandb_flag:
            wandb.init(project="classification-v1", name=f"{model.name}")
            # wandb.config.data_voxel_1 = test_dataset._voxel_size
            # wandb.config.data_voxel_2 = test_dataset._voxel_size_2
            wandb.config.loss_fn = args.loss_fn
            wandb.config.loading_best = True

        model.start_time = time()
        for epoch in range(1, 10000):
            res, avg_train_dist = model.do_epoch(train_data, optimizer, train=True)
            train_loss = res
            with torch.no_grad():
                res, avg_valid_dist = model.do_epoch(valid_data, optimizer, train=False)
            valid_loss = res
            print("avg train dist:", avg_train_dist)
            print("avg valid dist:", avg_valid_dist)
            if wandb_flag:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "avg_train_dist": avg_train_dist,
                        "avg_valid_dist": avg_valid_dist,
                    }
                )
            print(
                f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}"
            )
            # scheduler.step()
            best_valid_loss, updated = model.smart_save(
                epoch, valid_loss, best_valid_loss
            )
            if not updated:
                print(f"--> reloading best model from: {model.get_best_name()}")
                print(f"--> best loss was {best_valid_loss}")
                model.load_state_dict(torch.load(model.get_best_name()))
            if args.run_for and (time() - start_time) > args.run_for:
                print(f" --> Stopping training after {args.run_for} seconds")
                break


# For fast debug and development
train_dataset_dir = "./data/rlbench/train_roc_pan"
valid_dataset_dir = "./data/rlbench/valid_roc_pan"
# train_dataset_dir = "./data/rlbench/train_open"
# valid_dataset_dir = "./data/rlbench/valid_open"


if __name__ == "__main__":
    main()
