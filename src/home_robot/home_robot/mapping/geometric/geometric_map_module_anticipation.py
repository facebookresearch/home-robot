from typing import Tuple

import cv2
import kornia
import numpy as np
import pytorch3d.transforms as pt
import skimage.morphology
import torch
from einops import asnumpy, rearrange
from torch import Tensor
from torch.nn import functional as F

import home_robot.mapping.map_utils as mu
import home_robot.mapping.occant_utils.common as ocu
import home_robot.utils.depth as du
import home_robot.utils.pose as pu
import home_robot.utils.rotation as ru
from home_robot.mapping.geometric.geometric_map_module import GeometricMapModule
from home_robot.mapping.occant_utils.configs.defaults import get_cfg
from home_robot.mapping.occant_utils.occant_model import OccupancyAnticipator
from home_robot.mapping.semantic.constants import MapConstants as MC

# For debugging input and output maps - shows matplotlib visuals
debug_maps = False

EPS_MAPPER = 1e-8


class GeometricMapModuleWithAnticipation(GeometricMapModule):
    """
    This class is responsible for updating a dense 2D geometric map with
    the local and global maps and poses, and generating map features through anticipation.

    Map proposed in:
    Occupancy Anticipation for Efficient Exploration and Navigation
    https://arxiv.org/pdf/2008.09285.pdf
    https://github.com/facebookresearch/OccupancyAnticipation
    """

    def __init__(self, *args, occant_cfg_path, occant_ckpt_path, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.occant_cfg = get_cfg(occant_cfg_path)
        self.occant_model = OccupancyAnticipator(self.occant_cfg)
        self.load_model_weights(occant_ckpt_path)
        self.occant_model.eval()
        self.occant_model.to(device)
        self.model_device = device

    def load_model_weights(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["mapper_state_dict"]
        # Clean state_dict
        state_dict = {
            k.replace("mapper.projection_unit.main.", ""): v
            for k, v in state_dict.items()
            if k.startswith("mapper.projection_unit")
        }
        self.occant_model.load_state_dict(state_dict)
        print(
            "\n"
            + "=" * 10
            + " Successfully loaded OccAnt mapper weights "
            + "=" * 10
            + "\n"
        )

    def _update_local_map_and_pose(
        self,
        obs: Tensor,
        pose_delta: Tensor,
        prev_map: Tensor,
        prev_pose: Tensor,
        camera_pose: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth, segmentation) of shape
             (batch_size, 3 + 1, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape (batch_size, MC.NON_SEM_CHANNELS, M, M)
            prev_pose: previous pose of shape (batch_size, 3)
            camera_pose: current camera poseof shape (batch_size, 4, 4)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, MC.NON_SEM_CHANNELS, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        batch_size, obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype
        if camera_pose is not None:
            # TODO: make consistent between sim and real
            # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
            angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
            # For habitat - pull x angle
            # tilt = angles[:, -1]
            # For real robot
            tilt = angles[:, 1]

            # Get the agent pose
            # hab_agent_height = camera_pose[:, 1, 3] * 100
            agent_pos = camera_pose[:, :3, 3] * 100
            agent_height = agent_pos[:, 2]
        else:
            tilt = torch.zeros(batch_size)
            agent_height = self.agent_height

        depth = obs[:, 3, :, :].float()
        # Filter depth values
        depth[depth > self.max_depth] = 0
        depth[depth <= self.min_depth] = 0

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        if self.debug_mode:
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = obs[:, :3, :: self.du_scale, :: self.du_scale].permute(0, 2, 3, 1)
            xyz = point_cloud_t[0].reshape(-1, 3)
            rgb = rgb[0].reshape(-1, 3)
            print("-> Showing point cloud in camera coords")
            show_point_cloud(
                (xyz / 100.0).numpy(), (rgb / 255.0).numpy(), orig=np.zeros(3)
            )

        point_cloud_base_coords = du.transform_camera_view_t(
            point_cloud_t, agent_height, torch.rad2deg(tilt).numpy(), device
        )

        # Show the point cloud in base coordinates for debugging
        if self.debug_mode:
            print()
            print("------------------------------")
            print("agent angles =", angles)
            print("agent tilt   =", tilt)
            print("agent height =", agent_height, "preset =", self.agent_height)
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in base coords")
            show_point_cloud(
                (xyz / 100.0).numpy(), (rgb / 255.0).numpy(), orig=np.zeros(3)
            )

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
        )

        if self.debug_mode:
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in map coords")
            show_point_cloud(
                (xyz / 100.0).numpy(), (rgb / 255.0).numpy(), orig=np.zeros(3)
            )

        voxel_channels = 1

        init_grid = torch.zeros(
            batch_size,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            batch_size,
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )
        # Filter out zeroed depth values
        zero_mask = (
            depth[:, :: self.du_scale, :: self.du_scale] == 0
        )  # (batch_size, H, W)
        zero_mask = zero_mask.unsqueeze(1).expand(-1, voxel_channels, -1, -1)
        zero_mask = zero_mask.view(batch_size, voxel_channels, -1)
        feat[zero_mask] = 0

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)

        agent_height_proj = voxels[
            ..., self.min_mapped_height : self.max_mapped_height
        ].sum(4)
        # all_height_proj = voxels.sum(4)
        all_height_proj = voxels[..., 0 : self.max_mapped_height].sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = (
            (fp_map_pred / self.map_pred_threshold) >= 1.0
        ).float()  # (B, 1, H, W)
        fp_exp_pred = (
            (fp_exp_pred / self.exp_pred_threshold) >= 1.0
        ).float()  # (B, 1, H, W)

        ########################################################
        # Perform occupancy anticipation
        ########################################################
        # Dilate obstacles map to minimize domain gap
        # fp_map_pred = self.dilate_tensor(fp_map_pred, 3, iterations=2)
        # Filter obstacles to minimize domain gap
        # fp_map_pred = kornia.filters.median_blur(fp_map_pred, (5, 5))
        # -------------------------------------------------------
        # ------- Create observations for OccAnt mapper --------
        # -------------------------------------------------------
        # ---------------------- ego_map -----------------------
        # - channel 0 is one if occupied
        # - channel 1 is one if
        ego_map = torch.cat([fp_map_pred, fp_exp_pred], dim=1)  # (B, 2, H, W)
        ego_map = ego_map.to(self.model_device)
        # Resize ego_map to expected dimensions
        ego_map = F.interpolate(ego_map, size=self.occant_cfg.input_hw, mode="bilinear")
        # Transform coordinate systems
        ## Originally, agent is at the center-top of map looking down.
        ## OccAnt expects the agent to be at center-bottom of map looking up.
        ego_map = torch.flip(ego_map, [2])
        ego_map_rgb = self.convert_map2rgb(ego_map[0])
        ego_map_rgb_2 = self.convert_map2rgb(ego_map[0], enhance_obstacles=True)
        # ------------------------ rgb --------------------------
        rgb_obs = obs[:, :3, :, :]
        rgb_obs = ocu.process_image(
            rearrange(rgb_obs, "b c h w -> b h w c"),
            self.occant_cfg.image_mean,
            self.occant_cfg.image_std,
        )  # (B, C, H, W)
        rgb_obs = ocu.padded_resize(rgb_obs, self.occant_cfg.input_hw[0]).to(
            self.model_device
        )
        # -------------------------------------------------------
        # -------------------- Anticipation --------------------
        # -------------------------------------------------------
        ego_map_a = self.occant_model({"ego_map_gt": ego_map, "rgb": rgb_obs})[
            "occ_estimate"
        ]
        ego_map_a_rgb = self.convert_map2rgb(ego_map_a[0])
        # # Entropy-based filtering
        ego_map_a_ent = self.perform_entropy_filtering(ego_map_a)
        ego_map_a_ent_rgb = self.convert_map2rgb(ego_map_a_ent[0])
        # Resize ego_map back to original dimensions
        ego_map_a = F.interpolate(
            ego_map_a, size=fp_map_pred.shape[-2:], mode="bilinear"
        )
        # Transform coordinates
        ego_map_a = torch.flip(ego_map_a, [2])
        # Replace previous maps
        fp_map_pred = ego_map_a[:, 0:1, :, :].to(device)
        fp_exp_pred = ego_map_a[:, 1:2, :, :].to(device)
        # Visualize for debugging
        vis_rgb = np.concatenate(
            [ego_map_rgb, ego_map_rgb_2, ego_map_a_rgb, ego_map_a_ent_rgb], axis=1
        )
        cv2.imshow("Occant visualization", vis_rgb[..., ::-1])
        cv2.waitKey(30)
        ########################################################

        agent_view = torch.zeros(
            batch_size,
            MC.NON_SEM_CHANNELS,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, MC.OBSTACLE_MAP : MC.OBSTACLE_MAP + 1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, MC.EXPLORED_MAP : MC.EXPLORED_MAP + 1, y1:y2, x1:x2] = fp_exp_pred

        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)
        st_pose = current_pose.clone().detach()

        st_pose[:, :2] = -(
            (
                st_pose[:, :2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # Clamp to [0, 1] after transform agent view to map coordinates
        translated = torch.clamp(translated, min=0.0, max=1.0)

        # Perform map aggregation
        current_map = self.perform_map_aggregation(prev_map, translated)

        # Reset current location
        current_map[:, MC.CURRENT_LOCATION, :, :].fill_(0.0)
        curr_loc = current_pose[:, :2]
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).int()

        for e in range(batch_size):
            x, y = curr_loc[e]
            current_map[
                e,
                MC.CURRENT_LOCATION : MC.CURRENT_LOCATION + 2,
                y - 2 : y + 3,
                x - 2 : x + 3,
            ].fill_(1.0)

            # Set a disk around the agent to explored
            # This is around the current agent - we just sort of assume we know where we are
            try:
                radius = 10
                explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
                current_map[
                    e,
                    MC.EXPLORED_MAP,
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1,
                ][explored_disk == 1] = 1
                # Record the region the agent has been close to using a disc centered at the agent
                radius = self.been_close_to_radius // self.resolution
                been_close_disk = torch.from_numpy(skimage.morphology.disk(radius))
                current_map[
                    e,
                    MC.BEEN_CLOSE_MAP,
                    y - radius : y + radius + 1,
                    x - radius : x + radius + 1,
                ][been_close_disk == 1] = 1
            except IndexError:
                pass

        if debug_maps:
            import matplotlib.pyplot as plt

            explored = current_map[0, MC.EXPLORED_MAP].numpy()
            been_close = current_map[0, MC.BEEN_CLOSE_MAP].numpy()
            obs = current_map[0, MC.OBSTACLE_MAP].numpy()
            plt.subplot(231)
            plt.imshow(explored)
            plt.subplot(232)
            plt.imshow(been_close)
            plt.subplot(233)
            plt.imshow(been_close * explored)
            plt.subplot(234)
            plt.imshow(obs)
            plt.subplot(236)
            plt.imshow(been_close * obs)
            plt.show()
            breakpoint()

        if self.must_explore_close:
            current_map[:, MC.EXPLORED_MAP] = (
                current_map[:, MC.EXPLORED_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
            )
            current_map[:, MC.OBSTACLE_MAP] = (
                current_map[:, MC.OBSTACLE_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
            )

        return current_map, current_pose

    def dilate_tensor(self, x, size, iterations=1):
        """
        x - (bs, C, H, W)
        size - int / tuple of intes
        Assumes a kernel of ones with size 'size'.
        """
        if type(size) == int:
            padding = size // 2
        else:
            padding = tuple([v // 2 for v in size])
        for i in range(iterations):
            x = F.max_pool2d(x, size, stride=1, padding=padding)

        return x

    def perform_entropy_filtering(self, e):
        """
        Arguments:
            e - (B, 2, M, M) - predicted egomap
        """
        explored_mask = (e[:, 1] > self.occant_cfg.thresh_explored).float()
        log_e = torch.log(e + EPS_MAPPER)
        log_1_e = torch.log(1 - e + EPS_MAPPER)
        entropy = -e * log_e - (1 - e) * log_1_e
        entropy_mask = (entropy.mean(dim=1) < self.occant_cfg.thresh_entropy).float()
        explored_mask = explored_mask * entropy_mask
        e[:, 1] = explored_mask
        return e

    def perform_map_aggregation(self, prev_map, curr_map):
        """
        Arguments:
            prev_map - (B, N, M, M)
            curr_map - (B, N, M, M)
        """
        # maps = torch.cat((prev_map.unsqueeze(1), curr_map.unsqueeze(1)), 1)
        # agg_map, _ = torch.max(maps[:, :, : MC.NON_SEM_CHANNELS], 1)
        # Perform max-pool aggregation for non-anticipated channels
        agg_map = torch.zeros_like(prev_map)
        for i in [MC.CURRENT_LOCATION, MC.VISITED_MAP, MC.BEEN_CLOSE_MAP]:
            maps = torch.stack(
                (prev_map[:, i, :, :], curr_map[:, i, :, :]), dim=1
            )  # (B, 2, M, M)
            agg_map_i, _ = torch.max(maps, 1)
            agg_map[:, i] = agg_map_i
        # Perform moving-average aggregation for anticipated channels
        agg_map = self._moving_average_aggregation(curr_map, prev_map, agg_map)

        return agg_map

    # def _moving_average_aggregation(self, curr_map, prev_map, agg_map):
    #     """
    #     Arguments:
    #         prev_map - (B, N, M, M)
    #         curr_map - (B, N, M, M)
    #         agg_map - (B, N, M, M)
    #     """
    #     explored_mask = (curr_map[:, MC.EXPLORED_MAP] > self.occant_cfg.thresh_explored).float()
    #     unfilled_mask = (prev_map[:, MC.EXPLORED_MAP] == 0).float()
    #     # Previously unfilled and explored right now
    #     mask_0 = unfilled_mask * explored_mask
    #     # Previously filled and explored now
    #     mask_1 = (1 - unfilled_mask) * explored_mask
    #     beta = self.occant_cfg.AGGREGATOR.map_registration_momentum
    #     for i in [MC.OBSTACLE_MAP, MC.EXPLORED_MAP]:
    #         # Initially fill agg_map with prev_map values
    #         agg_map[:, i] = prev_map[:, i]
    #         # For unfilled regions, write the new map as it is
    #         agg_map[:, i] = agg_map[:, i] * (1 - mask_0) + curr_map[:, i] * mask_0
    #         # For filled regions, do a moving average
    #         ma_estimate_i = (curr_map[:, i] * (1 - beta) + prev_map[:, i] * beta) * mask_1
    #         agg_map[:, i] = agg_map[:, i] * (1 - mask_1) + ma_estimate_i * mask_1
    #     return agg_map

    def _moving_average_aggregation(self, curr_map, prev_map, agg_map):
        """
        Arguments:
            prev_map - (B, N, M, M)
            curr_map - (B, N, M, M)
            agg_map - (B, N, M, M)
        """
        beta = self.occant_cfg.AGGREGATOR.map_registration_momentum
        thresh_explored = self.occant_cfg.thresh_explored
        thresh_entropy = self.occant_cfg.thresh_entropy
        # Convert to OccAnt map registration format
        p_reg = curr_map[:, [MC.OBSTACLE_MAP, MC.EXPLORED_MAP]]
        m = prev_map[:, [MC.OBSTACLE_MAP, MC.EXPLORED_MAP]]
        ############################################################################
        # Entropy-moving-average map registration
        ############################################################################
        explored_mask = (p_reg[:, 1] > thresh_explored).float()
        log_p_reg = torch.log(p_reg + EPS_MAPPER)
        log_1_p_reg = torch.log(1 - p_reg + EPS_MAPPER)
        entropy = -p_reg * log_p_reg - (1 - p_reg) * log_1_p_reg
        entropy_mask = (entropy.mean(dim=1) < thresh_entropy).float()
        explored_mask = explored_mask * entropy_mask
        unfilled_mask = (m[:, 1] == 0).float()
        m_updated = m
        # For regions that are unfilled, write as it is
        mask = unfilled_mask * explored_mask
        mask = mask.unsqueeze(1)
        m_updated = m_updated * (1 - mask) + p_reg * mask
        # For regions that are filled, do a moving average
        mask = (1 - unfilled_mask) * explored_mask
        mask = mask.unsqueeze(1)
        p_reg_ma = (p_reg * (1 - beta) + m_updated * beta) * mask
        m_updated = m_updated * (1 - mask) + p_reg_ma * mask
        ############################################################################
        agg_map[:, MC.OBSTACLE_MAP] = m_updated[:, 0]
        agg_map[:, MC.EXPLORED_MAP] = m_updated[:, 1]
        return agg_map

    def convert_map2rgb(self, occ_map, enhance_obstacles=False):
        """
        Inputs:
            occ_map - (2, H, W) tensor with values between 0.0 to 1.0
        """
        exp_mask = asnumpy(occ_map[1] >= 0.5).astype(np.float32)
        occ_mask = asnumpy(occ_map[0] >= 0.5).astype(np.float32) * exp_mask
        free_mask = asnumpy(occ_map[0] < 0.5).astype(np.float32) * exp_mask
        if enhance_obstacles:
            kernel = np.ones((5, 5))
            occ_mask = cv2.dilate(occ_mask, kernel, iterations=1)
            free_mask[occ_mask == 1] = 0
            exp_mask[occ_mask == 1] = 1
        unk_mask = 1 - exp_mask

        occ_map_rgb = np.stack(
            [
                0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
                0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
                255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            ],
            axis=2,
        ).astype(
            np.uint8
        )  # (H, W, 3)

        return occ_map_rgb
