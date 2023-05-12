# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
from home_robot.utils.point_cloud import show_point_cloud
from home_robot.utils.rotation import get_angle_to_pos

HARDCODED_EXTENSION_OFFSET = 0.15
HARDCODED_YAW_OFFSET = 0.15


class HeuristicPlacePolicy(nn.Module):
    """
    Policy to place object on end receptacle using depth and point-cloud-based heuristics.
    """

    def __init__(self, config, device):
        super().__init__()
        self.timestep = 0
        self.config = config
        self.device = device
        self.visualize_point_clouds = False

    def get_receptacle_placement_point(
        self,
        obs: Observations,
        vis_inputs: None,
        arm_reachability_check: bool = False,
        visualize: bool = True,
    ):
        HEIGHT_OFFSET = 0.02

        goal_rec_mask = (
            obs.semantic == obs.task_observations["end_recep_goal"]
        ).astype(int)
        if visualize:
            cv2.imwrite(f"{self.end_receptacle}_semantic.png", goal_rec_mask * 255)

        if not goal_rec_mask.any():
            print("End receptacle not visible.")
            return False
        else:
            rgb_vis = obs.rgb
            goal_rec_depth = torch.tensor(
                obs.depth * goal_rec_mask, device=self.device, dtype=torch.float32
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

            if self.visualize_point_clouds:
                show_point_cloud(obs.xyz, obs.rgb / 255.0)

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

            # Whether or not I can extend the robot's arm in order to reach each point
            if arm_reachability_check:
                # filtering out unreachable points based on Y and Z coordinates of voxels
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

            y_values = pcd_base_coords[0, :, :, 2]

            non_zero_y_values = y_values[y_values != 0]

            if non_zero_y_values.numel() == 0:
                return False

            # extracting topmost voxels
            highest_points_mask = torch.bitwise_and(
                (y_values >= non_zero_y_values.max() - HEIGHT_OFFSET),
                (y_values <= non_zero_y_values.max()),
            ).to(torch.uint8)

            if visualize:
                highest_points_mask_vis = torch.stack(
                    [
                        highest_points_mask * 255,
                        highest_points_mask,
                        highest_points_mask,
                    ],
                    axis=-1,
                )  # for visualization
                alpha = 0.5
                rgb_vis = cv2.addWeighted(
                    rgb_vis, alpha, highest_points_mask_vis.cpu().numpy(), 1 - alpha, 0
                )
                cv2.imwrite(
                    f"{self.end_receptacle}_heights_vis.png", rgb_vis[:, :, ::-1]
                )

            highest_points_mask = torch.stack(
                [highest_points_mask, highest_points_mask, highest_points_mask],
                axis=-1,
            ).unsqueeze(0)

            pcd_base_coords_filtered = pcd_base_coords * highest_points_mask

            x_values = pcd_base_coords_filtered[..., 0]
            x_values = x_values[x_values != 0]
            x_min, x_max = x_values.min(), x_values.max()
            x_mean = (x_min + x_max) / 2

            z_values = pcd_base_coords_filtered[..., 1]
            z_values = z_values[z_values != 0]
            z_min, z_max = z_values.min(), z_values.max()
            z_mean = (z_min + z_max) / 2

            pcd_xz = pcd_base_coords_filtered[0, ..., 0:2]
            xz_mean = torch.tensor([x_mean, z_mean], device=self.device)
            xz_distances = torch.linalg.norm(pcd_xz - xz_mean, axis=-1)

            center_point = torch.where(xz_distances == xz_distances.min())
            center_x, center_y = center_point[0][0], center_point[1][0]
            center_voxel = pcd_base_coords_filtered[:, center_x, center_y, :]

            if visualize:
                rgb_vis = cv2.circle(
                    rgb_vis,
                    (center_y.item(), center_x.item()),
                    4,
                    (0, 255, 0),
                    thickness=2,
                )

                cv2.imwrite(
                    f'{obs.task_observations["goal_name"].split(" ")[-1]}.png',
                    rgb_vis[..., ::-1],
                )

            if vis_inputs is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis

            return center_voxel.cpu().numpy()[0], (center_x, center_y), vis_inputs

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

            if found:
                center_voxel, (center_x, center_y), vis_inputs = found
            else:
                print("Receptacle not visible. Abort.")
                action = DiscreteNavigationAction.STOP
                return action, vis_inputs

            center_voxel_trans = np.array(
                [center_voxel[1], center_voxel[2], center_voxel[0]]
            )

            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            self.initial_orient_num_turns = abs(delta_heading) // turn_angle
            self.orient_turn_direction = np.sign(delta_heading)

            # This gets the Y-coordiante of the center voxel
            # Base link to retracted arm - this is about 15 cm
            fwd_dist = (
                center_voxel[1] - STRETCH_STANDOFF_DISTANCE - HARDCODED_EXTENSION_OFFSET
            )
            # breakpoint()
            self.forward_steps = fwd_dist // fwd_step_size
            self.cam_arm_alignment_num_turns = np.round(90 / turn_angle)
            self.total_turn_and_forward_steps = (
                self.forward_steps
                + self.initial_orient_num_turns
                + self.cam_arm_alignment_num_turns
            )
            self.fall_wait_steps = 20
            breakpoint()

            print("-" * 20)
            print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
            print(f"Move forward for {self.forward_steps} steps.")
            print(
                f"Turn left to align camera and arm for {self.cam_arm_alignment_num_turns} steps."
            )

        print("-" * 20)
        print("Timestep", self.timestep)
        if self.timestep < self.initial_orient_num_turns:
            if self.orient_turn_direction == -1:
                print("Turning right to orient towards object")
                action = DiscreteNavigationAction.TURN_RIGHT
            if self.orient_turn_direction == +1:
                print("Turning left to orient towards object")
                action = DiscreteNavigationAction.TURN_LEFT
        elif self.timestep < self.initial_orient_num_turns + self.forward_steps:
            print("Moving forward")
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif self.timestep < self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.TURN_LEFT
            print("Turning left to align camera and arm")
        elif self.timestep == self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
            print("Aligning camera to arm")
        elif self.timestep == self.total_turn_and_forward_steps + 1:
            found = self.get_receptacle_placement_point(
                obs, vis_inputs, arm_reachability_check=True
            )
            if found is not False:
                center_voxel, (center_x, center_y), vis_inputs = found
            else:
                print("Receptacle not visible. Abort.")
                action = DiscreteNavigationAction.STOP
                return action, vis_inputs

            placement_height, placement_extension = center_voxel[2], center_voxel[1]

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift

            current_arm_ext = obs.joint[:4].sum()
            delta_arm_ext = (
                placement_extension
                - STRETCH_STANDOFF_DISTANCE
                - STRETCH_GRIPPER_OPEN
                - current_arm_ext
                + HARDCODED_EXTENSION_OFFSET
            )
            center_voxel_trans = np.array(
                [center_voxel[1], center_voxel[2], center_voxel[0]]
            )
            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            delta_gripper_yaw = delta_heading / 90 - HARDCODED_YAW_OFFSET

            print("Delta arm extension:", delta_arm_ext)
            print("Delta arm lift:", delta_arm_lift)
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
            print("Desnapping object")
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif (
            self.timestep
            <= self.total_turn_and_forward_steps + 2 + self.fall_wait_steps
        ):
            print("Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            print("Stopping")
            action = DiscreteNavigationAction.STOP

        self.timestep += 1
        return action, vis_inputs
