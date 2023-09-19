# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

import home_robot.utils.depth as du
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.manipulation.heuristic_place_policy import HeuristicPlacePolicy
from home_robot.utils.image import smooth_mask
from home_robot.utils.rotation import get_angle_to_pos

RETRACTED_ARM_APPROX_LENGTH = 0.15
STRETCH_STANDOFF_DISTANCE = 0.6
STRETCH_GRASP_DISTANCE = 0.4
STRETCH_EXTENSION_OFFSET = 0.0
ANGLE_ADJUSTMENT = -0.01


class HeuristicPickPolicy(HeuristicPlacePolicy):
    """
    Heuristic policy for picking objects.
    Mainly used for visualizing the agent's arm reaching the object.
    First determines the pick point using object point cloud, then turns to orient towards the object, then moves the arm to the pick point and snaps the object.
    """

    def __init__(
        self, config, device, debug_visualize_xyz: bool = False, verbose: bool = False
    ):
        self.timestep = 0
        self.verbose = verbose
        super().__init__(config, device, debug_visualize_xyz)

    def get_object_pick_point(
        self,
        obs: Observations,
        vis_inputs: Optional[Dict] = None,
        arm_reachability_check: bool = False,
        visualize: bool = True,
    ):
        goal_object_mask = (
            obs.semantic
            == obs.task_observations["object_goal"] * du.valid_depth_mask(obs.depth)
        ).astype(np.uint8)
        # Get dilated, then eroded mask (for cleanliness)
        goal_object_mask = smooth_mask(
            goal_object_mask, self.erosion_kernel, num_iterations=5
        )[1]
        # Convert to booleans
        goal_object_mask = goal_object_mask.astype(bool)

        if visualize:
            cv2.imwrite(f"{self.object_name}_semantic.png", goal_object_mask * 255)

        if not goal_object_mask.any():
            if self.verbose:
                print("Goal object not visible!")
            return None
        else:
            pcd_base_coords = self.get_target_point_cloud_base_coords(
                obs, goal_object_mask, arm_reachability_check
            )

            pcd_base_coords = pcd_base_coords.cpu().numpy()
            flat_array = pcd_base_coords.reshape(-1, 3)
            index = flat_array[:, 2].argmax()
            best_voxel = pcd_base_coords[
                index // pcd_base_coords.shape[-2], index % pcd_base_coords.shape[-2]
            ]
            rgb_vis = obs.rgb.copy()
            rgb_vis_tmp = cv2.circle(
                rgb_vis,
                (index % pcd_base_coords.shape[-2], index // pcd_base_coords.shape[-2]),
                4,
                (0, 255, 0),
                thickness=2,
            )
            if vis_inputs is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis_tmp
            return best_voxel, vis_inputs

    def generate_plan(
        self, obs: Observations, vis_inputs: Optional[Dict] = None
    ) -> None:
        """Hardcode the following plan:
        1. Find a grasp point.
        2. Turn to orient towards the object.
        3. Raise the arm.
        4. Move the arm to the object.
        5. Snap the object.
        6. Raise the arm.
        7. Close gripper."""
        self.du_scale = 1
        self.object_name = obs.task_observations["goal_name"].split(" ")[1]
        found = self.get_object_pick_point(obs, vis_inputs)
        if found is None:
            # if not found, we retry after tilt is lowered. Otherwise, we just snap the object
            self.orient_turn_angle = 0
            fwd_dist = 0
            self.fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
            self.t_turn_to_orient = np.inf
            self.t_move_to_reach = np.inf
            self.t_manip_mode = 0
            self.t_turn_to_orient_post_manip_mode = 1
            self.t_relative_back = np.inf
            self.t_relative_standoff = np.inf
            self.t_relative_grasp = np.inf
            self.t_relative_snap_object = 0
            self.t_start_pick = np.inf
        else:
            self.grasp_voxel, vis_inputs = found

            center_voxel_trans = np.array(
                [
                    self.grasp_voxel[1],
                    self.grasp_voxel[2],
                    self.grasp_voxel[0],
                ]
            )

            delta_heading = get_angle_to_pos(center_voxel_trans)
            self.orient_turn_angle = delta_heading
            # This gets the Y-coordiante of the center voxel
            # Base link to retracted arm - this is about 15 cm
            fwd_dist = self.grasp_voxel[1]
            self.fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
            self.t_turn_to_orient = 0
            self.t_move_to_reach = 1
            self.t_manip_mode = 2
            self.t_turn_to_orient_post_manip_mode = 3
            # timesteps relative to the time when orientation finishes
            self.t_relative_back = 0
            self.t_relative_standoff = 1
            self.t_relative_grasp = 2
            self.t_relative_snap_object = 3
            self.t_start_pick = np.inf
            if self.verbose:
                print("-" * 20)

    def get_action(
        self, obs: Observations, vis_inputs: Optional[Dict] = None
    ) -> Tuple[
        Union[
            ContinuousFullBodyAction,
            ContinuousNavigationAction,
            DiscreteNavigationAction,
        ],
        Dict,
    ]:
        """Get the action to execute at the current timestep using the plan generated in generate_plan.
        Before actual picking starts (i.e. before t_start_pick), the agent turns and moves to orient towards the pick point.
        Recalibrates the pick point after switching to manipulation mode.
        After t_start_pick, the agent moves the arm to the pick point and snaps the object.
        """
        action = None
        if self.timestep == self.t_turn_to_orient:
            # at first turn to face the object and move forward
            action = ContinuousNavigationAction([0, 0, self.orient_turn_angle])
            self.orient_turn_angle = 0
        elif self.timestep == self.t_move_to_reach:
            action = ContinuousNavigationAction([self.fwd_dist, 0, 0])
        elif self.timestep == self.t_manip_mode:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif self.timestep == self.t_turn_to_orient_post_manip_mode:
            grasp_voxel = self.get_object_pick_point(obs, vis_inputs)
            # recalibrate the grasp voxel (since the agent may have moved a bit and is looking down)
            if grasp_voxel is not None:
                self.grasp_voxel, vis_inputs = grasp_voxel
                center_voxel_trans = np.array(
                    [
                        self.grasp_voxel[1],
                        self.grasp_voxel[2],
                        self.grasp_voxel[0],
                    ]
                )
                self.orient_turn_angle = (
                    get_angle_to_pos(center_voxel_trans) + ANGLE_ADJUSTMENT
                )
            if self.orient_turn_angle == 0:
                self.t_start_pick = self.timestep
            else:
                action = ContinuousNavigationAction([0, 0, self.orient_turn_angle])
                self.orient_turn_angle = 0
                self.t_start_pick = self.timestep + 1
                self.t_relative_back = 0
                self.t_relative_standoff = 1
                self.t_relative_grasp = 2
                self.t_relative_snap_object = 3
        if action is not None:
            return action, vis_inputs

        if self.timestep == self.t_start_pick + self.t_relative_back:
            # final recalibration of the grasp voxel
            grasp_voxel = self.get_object_pick_point(obs, vis_inputs)
            if grasp_voxel is not None:
                self.grasp_voxel, vis_inputs = grasp_voxel
            standoff_lift = np.min(
                [1.1, self.grasp_voxel[2] + STRETCH_STANDOFF_DISTANCE]
            )
            current_arm_lift = obs.joint[4]
            delta_arm_lift = standoff_lift - current_arm_lift
            joints = np.array([0] * 4 + [delta_arm_lift] + [0] * 5)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_start_pick + self.t_relative_standoff:
            target_extension = self.grasp_voxel[1]
            current_arm_ext = obs.joint[:4].sum()
            delta_arm_ext = (
                target_extension - current_arm_ext - STRETCH_EXTENSION_OFFSET
            )
            joints = np.array([delta_arm_ext] + [0] * 9)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_start_pick + self.t_relative_grasp:
            grasp_lift = np.min([1.1, self.grasp_voxel[2] + STRETCH_GRASP_DISTANCE])
            current_arm_lift = obs.joint[4]
            delta_arm_lift = grasp_lift - current_arm_lift
            joints = np.array([0] * 4 + [delta_arm_lift] + [0] * 5)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_start_pick + self.t_relative_snap_object:
            # snap to pick the object
            if self.verbose:
                print("[Pick] Snapping object")
            action = DiscreteNavigationAction.SNAP_OBJECT
        else:
            if self.verbose:
                print("[Pick] Stopping")
            action = DiscreteNavigationAction.STOP
        return action, vis_inputs

    def forward(self, obs: Observations, vis_inputs: Optional[Dict] = None):
        self.timestep = self.timestep

        if self.timestep == 0:
            self.generate_plan(obs, vis_inputs)
        if self.verbose:
            print("-" * 20)
            print("Timestep", self.timestep)
        action, vis_inputs = self.get_action(obs, vis_inputs)

        self.timestep += 1
        return action, vis_inputs
