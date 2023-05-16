# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh.transformations as tra

import home_robot.utils.depth as du
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.motion.stretch import STRETCH_GRIPPER_OPEN, STRETCH_STANDOFF_DISTANCE
from home_robot.utils.point_cloud import valid_depth_mask

# from home_robot.utils.point_cloud import show_point_cloud
from home_robot.utils.rotation import get_angle_to_pos

RETRACTED_ARM_APPROX_LENGTH = 0.15
HARDCODED_ARM_EXTENSION_OFFSET = 0.15
HARDCODED_YAW_OFFSET = 0.25


class HeuristicPlacePolicy(nn.Module):
    """
    Policy to place object on end receptacle using depth and point-cloud-based heuristics.
    """

    def __init__(self, config, device, debug_visualize_xyz: bool = True):
        super().__init__()
        self.timestep = 0
        self.config = config
        self.device = device
        self.debug_visualize_xyz = debug_visualize_xyz

    def get_receptacle_placement_point(
        self,
        obs: Observations,
        vis_inputs: None,
        arm_reachability_check: bool = False,
        visualize: bool = True,
    ):
        NUM_POINTS_TO_SAMPLE = 50  # number of points to sample from receptacle point cloud to find best placement point
        SLAB_PADDING = 0.2  # x/y padding around randomly selected points
        SLAB_HEIGHT_THRESHOLD = 0.015  # 1cm above and below, i.e. 2cm overall
        ALPHA_VIS = 0.5

        goal_rec_mask = (
            obs.semantic
            == obs.task_observations["end_recep_goal"] * valid_depth_mask(obs.depth)
        ).astype(bool)
        if visualize:
            cv2.imwrite(f"{self.end_receptacle}_semantic.png", goal_rec_mask * 255)

        if not goal_rec_mask.any():
            print("End receptacle not visible.")
            return None
        else:
            rgb_vis = obs.rgb
            goal_rec_depth = torch.tensor(
                obs.depth, device=self.device, dtype=torch.float32
            ).unsqueeze(0)

            camera_matrix = du.get_camera_matrix(
                self.config.ENVIRONMENT.frame_width,
                self.config.ENVIRONMENT.frame_height,
                self.config.ENVIRONMENT.hfov,
            )
            # Get object point cloud in camera coordinates
            pcd_camera_coords = du.get_point_cloud_from_z_t(
                goal_rec_depth, camera_matrix, self.device, scale=self.du_scale
            )

            # get point cloud in base coordinates
            camera_pose = np.expand_dims(obs.camera_pose, 0)
            angles = [tra.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
            tilt = angles[0][1]  # [0][1]

            # Agent height comes from the environment config
            agent_height = torch.tensor(
                self.config.ENVIRONMENT.camera_height, device=self.device
            )

            # Object point cloud in base coordinates
            pcd_base_coords = du.transform_camera_view_t(
                pcd_camera_coords, agent_height, np.rad2deg(tilt), self.device
            )

            if self.debug_visualize_xyz:
                # Remove invalid points from the mask
                xyz = (
                    pcd_base_coords[0]
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)[goal_rec_mask.reshape(-1), :]
                )
                from home_robot.utils.point_cloud import show_point_cloud

                rgb = (obs.rgb).reshape(-1, 3) / 255.0
                show_point_cloud(xyz, rgb, orig=np.zeros(3))

            # Whether or not I can extend the robot's arm in order to reach each point
            if arm_reachability_check:
                # filtering out unreachable points based on Y and Z coordinates of voxels (Z is up)
                height_reachable_mask = (pcd_base_coords[0, :, :, 2] < agent_height).to(
                    int
                )
                height_reachable_mask = torch.stack(
                    [height_reachable_mask] * 3, axis=-1
                )
                pcd_base_coords = pcd_base_coords * height_reachable_mask

                length_reachable_mask = (pcd_base_coords[0, :, :, 1] < agent_height).to(
                    int
                )
                length_reachable_mask = torch.stack(
                    [length_reachable_mask] * 3, axis=-1
                )
                pcd_base_coords = pcd_base_coords * length_reachable_mask

            non_zero_mask = torch.stack(
                [torch.from_numpy(goal_rec_mask).to(self.device)] * 3, axis=-1
            )
            pcd_base_coords = pcd_base_coords * non_zero_mask

            ## randomly sampling NUM_POINTS_TO_SAMPLE of receptacle point cloud – to choose for placement

            reachable_point_cloud = pcd_base_coords[0].cpu().numpy()
            flat_array = reachable_point_cloud.reshape(-1, 3)

            # find the indices of the non-zero elements in the first two dimensions of the matrix
            nonzero_indices = np.nonzero(flat_array[:, :2].any(axis=1))[0]
            # create a list of tuples containing the non-zero indices in the first two dimensions
            nonzero_tuples = [
                (
                    index // reachable_point_cloud.shape[-2],
                    index % reachable_point_cloud.shape[-2],
                )
                for index in nonzero_indices
            ]
            # select a random subset of the non-zero indices
            random_indices = random.sample(
                nonzero_tuples, min(NUM_POINTS_TO_SAMPLE, len(nonzero_tuples))
            )

            x_values = pcd_base_coords[0, :, :, 0]
            y_values = pcd_base_coords[0, :, :, 1]
            z_values = pcd_base_coords[0, :, :, 2]

            max_surface_points = 0
            # max_height = 0
            max_surface_mask, best_voxel_ind, best_voxel = None, None, None

            ## iterating through all randomly selected voxels and choosing one with most XY neighboring surface area within some height threshold

            for ind in random_indices:
                sampled_voxel = pcd_base_coords[0, ind[0], ind[1]]
                sampled_voxel_x, sampled_voxel_y, sampled_voxel_z = (
                    sampled_voxel[0],
                    sampled_voxel[1],
                    sampled_voxel[2],
                )

                # sampling plane of pcd voxels around randomly selected voxel (with height tolerance)
                slab_points_mask_x = torch.bitwise_and(
                    (x_values >= sampled_voxel_x - SLAB_PADDING),
                    (x_values <= sampled_voxel_x + SLAB_PADDING),
                )
                slab_points_mask_y = torch.bitwise_and(
                    (y_values >= sampled_voxel_y - SLAB_PADDING),
                    (y_values <= sampled_voxel_y + SLAB_PADDING),
                )
                slab_points_mask_z = torch.bitwise_and(
                    (z_values >= sampled_voxel_z - SLAB_HEIGHT_THRESHOLD),
                    (z_values <= sampled_voxel_z + SLAB_HEIGHT_THRESHOLD),
                )

                slab_points_mask = torch.bitwise_and(
                    slab_points_mask_x, slab_points_mask_y
                ).to(torch.uint8)
                slab_points_mask = torch.bitwise_and(
                    slab_points_mask, slab_points_mask_z
                ).to(torch.uint8)

                # ALTERNATIVE: choose slab with maximum (area x height) product

                # slab_points_mask_stacked = torch.stack(
                #     [
                #         slab_points_mask * 255,
                #         slab_points_mask,
                #         slab_points_mask,
                #     ],
                #     axis=-1,
                # )
                # height = (slab_points_mask_stacked * pcd_base_coords)[..., 2].max()
                # if slab_points_mask.sum() * height >= max_surface_points * max_height:
                if slab_points_mask.sum() >= max_surface_points:
                    max_surface_points = slab_points_mask.sum()
                    max_surface_mask = slab_points_mask
                    # max_height = height
                    best_voxel_ind = ind
                    best_voxel = sampled_voxel

            slab_points_mask_vis = torch.stack(
                [
                    max_surface_mask * 255,
                    max_surface_mask,
                    max_surface_mask,
                ],
                axis=-1,
            )  # for visualization
            rgb_vis_tmp = cv2.addWeighted(
                rgb_vis, ALPHA_VIS, slab_points_mask_vis.cpu().numpy(), 1 - ALPHA_VIS, 0
            )

            rgb_vis_tmp = cv2.circle(
                rgb_vis_tmp,
                (best_voxel_ind[1], best_voxel_ind[0]),
                4,
                (0, 255, 0),
                thickness=2,
            )

            if vis_inputs is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis_tmp

            if self.debug_visualize_xyz:
                show_point_cloud(
                    pcd_base_coords[0].cpu().numpy(),
                    rgb=obs.rgb / 255.0,
                    orig=best_voxel.cpu().numpy(),
                )

            return best_voxel.cpu().numpy(), vis_inputs

    def forward(self, obs: Observations, vis_inputs=None):
        """
        1. Get estimate of point on receptacle to place object on.
        2. Orient towards it.
        3. Move forward to get close to it.
        4. Rotate 90º to have arm face the object. Then rotate camera to face arm.
        5. (again) Get estimate of point on receptacle to place object on.
        6. With camera, arm, and object (hopefully) aligned, set arm lift and
        extension based on point estimate from 4.
        """

        self.timestep = self.timestep
        turn_angle = self.config.ENVIRONMENT.turn_angle
        fwd_step_size = self.config.ENVIRONMENT.forward

        if self.timestep == 0:
            self.du_scale = 1  # TODO: working with full resolution for now
            self.end_receptacle = obs.task_observations["goal_name"].split(" ")[-1]
            found = self.get_receptacle_placement_point(obs, vis_inputs)

            if found is not None:
                self.placement_voxel, vis_inputs = found
            else:
                print("Receptacle not visible. Abort.")
                action = DiscreteNavigationAction.STOP
                return action, vis_inputs

            center_voxel_trans = np.array(
                [
                    self.placement_voxel[1],
                    self.placement_voxel[2],
                    self.placement_voxel[0],
                ]
            )

            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            self.initial_orient_num_turns = abs(delta_heading) // turn_angle
            self.orient_turn_direction = np.sign(delta_heading)
            # This gets the Y-coordiante of the center voxel
            # Base link to retracted arm - this is about 15 cm
            fwd_dist = (
                self.placement_voxel[1]
                - STRETCH_STANDOFF_DISTANCE
                - RETRACTED_ARM_APPROX_LENGTH
            )

            fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
            self.forward_steps = fwd_dist // fwd_step_size
            self.total_turn_and_forward_steps = (
                self.forward_steps + self.initial_orient_num_turns
            )
            self.fall_wait_steps = 20

            print("-" * 20)
            print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
            print(f"Move forward for {self.forward_steps} steps.")

        print("-" * 20)
        print("Timestep", self.timestep)
        if self.timestep < self.initial_orient_num_turns:
            if self.orient_turn_direction == -1:
                print("[Placement] Turning right to orient towards object")
                action = DiscreteNavigationAction.TURN_RIGHT
            if self.orient_turn_direction == +1:
                print("[Placement] Turning left to orient towards object")
                action = DiscreteNavigationAction.TURN_LEFT
        elif self.timestep < self.total_turn_and_forward_steps:
            print("[Placement] Moving forward")
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif self.timestep == self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
            print("[Placement] Aligning camera to arm")
        elif self.timestep == self.total_turn_and_forward_steps + 1:
            print("[Placement] Move arm into position")
            placement_height, placement_extension = (
                self.placement_voxel[2],
                self.placement_voxel[1],
            )

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift

            current_arm_ext = obs.joint[:4].sum()
            delta_arm_ext = (
                placement_extension
                - STRETCH_STANDOFF_DISTANCE
                - RETRACTED_ARM_APPROX_LENGTH
                - current_arm_ext
                + HARDCODED_ARM_EXTENSION_OFFSET
            )
            center_voxel_trans = np.array(
                [
                    self.placement_voxel[1],
                    self.placement_voxel[2],
                    self.placement_voxel[0],
                ]
            )
            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            delta_gripper_yaw = delta_heading / 90 - HARDCODED_YAW_OFFSET

            print("[Placement] Delta arm extension:", delta_arm_ext)
            print("[Placement] Delta arm lift:", delta_arm_lift)
            joints = (
                [delta_arm_ext]
                + [0] * 3
                + [delta_arm_lift]
                + [delta_gripper_yaw]
                + [0] * 4
            )
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.total_turn_and_forward_steps + 2:
            # desnap to drop the object
            print("[Placement] Desnapping object")
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif (
            self.timestep
            <= self.total_turn_and_forward_steps + 2 + self.fall_wait_steps
        ):
            print("[Placement] Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            print("[Placement] Stopping")
            action = DiscreteNavigationAction.STOP

        self.timestep += 1
        return action, vis_inputs
