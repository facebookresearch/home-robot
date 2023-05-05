from enum import IntEnum
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import trimesh.transformations as tra
from habitat.tasks.rearrange.utils import get_angle_to_pos

import home_robot.utils.depth as du
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations


class Skill(IntEnum):
    NAV_TO_OBJ = 0
    ORIENT_OBJ = 1
    PICK = 2
    NAV_TO_REC = 3
    PLACE = 4


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.orient_start_step = None
        self.is_pick_done = None
        self.place_done = None
        self.gaze_agent = None
        if config.AGENT.SKILLS.PICK.type == "gaze":
            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.PICK,
                device_id=device_id,
                obs_spaces=obs_spaces,
                action_spaces=action_spaces,
            )
        self.skip_nav_to_obj = config.AGENT.skip_nav_to_obj
        self.skip_nav_to_rec = config.AGENT.skip_nav_to_rec
        self.skip_place = config.AGENT.skip_place
        self.skip_pick = config.AGENT.skip_pick
        self.skip_orient_obj = config.AGENT.skip_orient_obj
        self.config = config

    def _get_vis_inputs(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        return {
            "semantic_frame": obs.task_observations["semantic_frame"],
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
            "curr_skill": Skill(self.states[0].item()).name,
        }

    def reset_vectorized(self, episodes):
        """Initialize agent state."""
        super().reset_vectorized()
        self.planner.set_vis_dir(
            episodes[0].scene_id.split("/")[-1].split(".")[0], episodes[0].episode_id
        )
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.orient_start_step = torch.tensor([0] * self.num_environments)
        self.is_pick_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)

    def get_receptacle_placement_point(
        self,
        obs: Observations,
        arm_reachability_check: bool = False,
        visualize: bool = False,
    ):
        MAX_ARM_LENGTH = 1.5
        HEIGHT_DIFF_THRESHOLD = 0.02

        goal_rec_mask = (obs.semantic == 3).astype(int)
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
            pcd_camera_coords = du.get_point_cloud_from_z_t(
                goal_rec_depth, camera_matrix, self.device, scale=self.du_scale
            )

            # get point cloud in base coordinates
            camera_pose = np.expand_dims(obs.camera_pose, 0)
            angles = [tra.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
            tilt = angles[0][1]  # [0][1]

            agent_height = torch.tensor(
                self.config.ENVIRONMENT.camera_height, device=self.device
            )

            pcd_base_coords = du.transform_camera_view_t(
                pcd_camera_coords, agent_height, np.rad2deg(tilt), self.device
            )

            if arm_reachability_check:
                # filtering out unreachable points by calculating distance from camera
                distances = torch.linalg.norm(pcd_camera_coords, axis=-1)
                reachable_mask = (distances < MAX_ARM_LENGTH).to(int)
                reachable_mask = torch.stack([reachable_mask] * 3, axis=-1)
                pcd_base_coords = pcd_base_coords * reachable_mask

            non_zero_mask = torch.stack(
                [torch.from_numpy(goal_rec_mask).to(self.device)] * 3, axis=-1
            )
            pcd_base_coords = pcd_base_coords * non_zero_mask

            y_values = pcd_base_coords[0, :, :, 2]

            non_zero_y_values = y_values[y_values != 0]

            # extracting topmost voxels
            highest_points_mask = torch.bitwise_and(
                (y_values >= non_zero_y_values.max() - HEIGHT_DIFF_THRESHOLD),
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

            return center_voxel.cpu().numpy()[0], (center_x, center_y)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int, episode):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.orient_start_step[e] = 0
        self.is_pick_done[e] = 0
        self.place_done[e] = 0
        super().reset_vectorized_for_env(e)
        self.planner.set_vis_dir(
            episode.scene_id.split("/")[-1].split(".")[0], episode.episode_id
        )
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)

    def _switch_to_next_skill(self, e: int):
        """Switch to the next skill for environment e."""
        skill = self.states[e]
        if skill == Skill.NAV_TO_OBJ:
            self.states[e] = Skill.ORIENT_OBJ
            self.orient_start_step[e] = self.timesteps[e]
        elif skill == Skill.ORIENT_OBJ:
            self.states[e] = Skill.PICK
        elif skill == Skill.PICK:
            self.timesteps_before_goal_update[0] = 0
            self.states[e] = Skill.NAV_TO_REC
        elif skill == Skill.NAV_TO_REC:
            self.place_start_step[e] = self.timesteps[e]
            self.states[e] = Skill.PLACE
        elif skill == Skill.PLACE:
            self.place_done[0] = 1

    def _modular_nav(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Any]:
        action, info = super().act(obs)
        self.timesteps[0] -= 1  # objectnav agent increments timestep
        info["timestep"] = self.timesteps[0]
        info["curr_skill"] = Skill(self.states[0].item()).name
        if action == DiscreteNavigationAction.STOP:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            self._switch_to_next_skill(e=0)
        return action, info

    def _modular_place(self, obs: Observations):
        """
        1. Get estimate of point on receptacle to place object on.
        2. Orient towards it.
        3. Move forward to get close to it.
        4. Rotate 90º to have arm face the object. Then rotate camera to face arm.
        5. (again) Get estimate of point on receptacle to place object on.
        6. With camera, arm, and object (hopefully) aligned, set arm lift and
        extension based on point estimate from 4.
        """
        RETRACTED_ARM_LENGTH = 0.2
        GRIPPER_LENGTH = 0.12

        place_step = self.timesteps[0] - self.place_start_step[0]
        turn_angle = self.config.ENVIRONMENT.turn_angle
        fwd_step_size = (
            self.config.habitat.simulator.forward_step_size
        )  # or self.config.AGENT.SKILLS.PICK.max_forward

        if place_step == 0:
            self.du_scale = 1  # TODO: working with full resolution for now
            self.end_receptacle = obs.task_observations["goal_name"].split(" ")[-1]
            found = self.get_receptacle_placement_point(obs)

            if found is not False:
                center_voxel, (center_x, center_y) = found
            else:
                print("Receptacle not visible. Abort.")
                action = DiscreteNavigationAction.STOP
                return action

            center_voxel_trans = np.array(
                [center_voxel[1], center_voxel[2], center_voxel[0]]
            )

            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            self.initial_orient_num_turns = abs(delta_heading) // turn_angle
            self.orient_turn_direction = np.sign(delta_heading)

            fwd_dist = center_voxel[1] - RETRACTED_ARM_LENGTH - GRIPPER_LENGTH

            self.forward_steps = fwd_dist // fwd_step_size
            self.cam_arm_alignment_num_turns = np.round(90 / turn_angle)
            self.total_turn_and_forward_steps = (
                self.forward_steps
                + self.initial_orient_num_turns
                + self.cam_arm_alignment_num_turns
            )
            self.fall_wait_steps = 20

            print("-" * 20)
            print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
            print(f"Move forward for {self.forward_steps} steps.")
            print(
                f"Turn left to align camera and arm for {self.cam_arm_alignment_num_turns} steps."
            )

        print("-" * 20)
        print("Timestep", place_step.item())
        if place_step < self.initial_orient_num_turns:
            if self.orient_turn_direction == -1:
                print("Turning right to orient towards object")
                action = DiscreteNavigationAction.TURN_RIGHT
            if self.orient_turn_direction == +1:
                print("Turning left to orient towards object")
                action = DiscreteNavigationAction.TURN_LEFT
        elif place_step < self.initial_orient_num_turns + self.forward_steps:
            print("Moving forward")
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step < self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.TURN_LEFT
            print("Turning left to align camera and arm")
        elif place_step == self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
            print("Aligning camera to arm")
        elif place_step == self.total_turn_and_forward_steps + 1:
            found = self.get_receptacle_placement_point(
                obs, arm_reachability_check=True
            )
            if found is not False:
                center_voxel, (center_x, center_y) = found
            else:
                print("Receptacle not visible. Abort.")
                action = DiscreteNavigationAction.STOP
                return action

            placement_height, placement_extension = center_voxel[2], center_voxel[1]

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift

            current_arm_ext = obs.joint[:4].sum() / (0.13 * 4)
            delta_arm_ext = (
                placement_extension
                - RETRACTED_ARM_LENGTH
                - GRIPPER_LENGTH
                - current_arm_ext
            )

            delta_gripper_yaw = 0

            print("Delta arm extension:", delta_arm_ext)
            print("Delta arm lift:", delta_arm_lift)

            action = {
                "arm_action": [delta_arm_ext]
                + [delta_arm_lift]
                + [delta_gripper_yaw]
                + [0] * 4,
                "grip_action": [1],
            }
        elif place_step == self.total_turn_and_forward_steps + 2:
            # desnap to drop the object
            print("Desnapping object")
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= self.total_turn_and_forward_steps + 2 + self.fall_wait_steps:
            print("Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            print("Stopping")
            action = DiscreteNavigationAction.STOP
        return action

    def _hardcoded_place(self):
        """Hardcoded place skill execution
        Orients the agent's arm and camera towards the recetacle, extends arm and releases the object"""
        place_step = self.timesteps[0] - self.place_start_step[0]
        turn_angle = self.config.ENVIRONMENT.turn_angle
        forward_steps = 0
        fall_steps = 20
        num_turns = np.round(90 / turn_angle)
        forward_and_turn_steps = forward_steps + num_turns
        if place_step <= forward_steps:
            # for experimentation (TODO: Remove. ideally nav should drop us close)
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step <= forward_and_turn_steps:
            # first orient
            action = DiscreteNavigationAction.TURN_LEFT
        elif place_step == forward_and_turn_steps + 1:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif place_step == forward_and_turn_steps + 2:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif place_step == forward_and_turn_steps + 3:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= forward_and_turn_steps + 3 + fall_steps:
            # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        elif place_step == forward_and_turn_steps + fall_steps + 4:
            action = DiscreteNavigationAction.STOP
        return action

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        vis_inputs = self._get_vis_inputs(obs)
        turn_angle = self.config.ENVIRONMENT.turn_angle

        self.timesteps[0] += 1
        if self.states[0] == Skill.NAV_TO_OBJ:
            if self.skip_nav_to_obj:
                self._switch_to_next_skill(e=0)
            elif self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "modular":
                return self._modular_nav(obs)
            else:
                raise NotImplementedError
        if self.states[0] == Skill.ORIENT_OBJ:
            num_turns = np.round(90 / turn_angle)
            orient_step = self.timesteps[0] - self.orient_start_step[0]
            if self.skip_orient_obj:
                self._switch_to_next_skill(e=0)
            elif orient_step <= num_turns:
                return DiscreteNavigationAction.TURN_LEFT, vis_inputs
            elif orient_step == num_turns + 1:
                self._switch_to_next_skill(e=0)
                return DiscreteNavigationAction.MANIPULATION_MODE, vis_inputs
        if self.states[0] == Skill.PICK:
            if self.skip_pick:
                self._switch_to_next_skill(e=0)
            elif self.is_pick_done[0]:
                self._switch_to_next_skill(e=0)
                self.is_pick_done[0] = 0
                return DiscreteNavigationAction.NAVIGATION_MODE, vis_inputs
            elif self.config.AGENT.SKILLS.PICK.type == "gaze":
                action, term = self.gaze_agent.act(obs)
                if term:
                    action = (
                        {}
                    )  # TODO: update after simultaneous gripping/motion is supported
                    action["grip_action"] = [1]  # grasp the object when gaze is done
                    self.is_pick_done[0] = 1
                return action, vis_inputs
            elif self.config.AGENT.SKILLS.PICK.type == "oracle":
                self.is_pick_done[0] = 1
                return DiscreteNavigationAction.SNAP_OBJECT, vis_inputs
            else:
                raise NotImplementedError
        if self.states[0] == Skill.NAV_TO_REC:
            if self.skip_nav_to_rec:
                self._switch_to_next_skill(e=0)
            elif self.config.AGENT.SKILLS.NAV_TO_REC.type == "modular":
                return self._modular_nav(obs)
            else:
                raise NotImplementedError
        if self.states[0] == Skill.PLACE:
            if self.skip_place:
                return DiscreteNavigationAction.STOP, vis_inputs
            elif self.config.AGENT.SKILLS.PLACE.type == "hardcoded":
                action = self._hardcoded_place()
                return action, vis_inputs
            elif self.config.AGENT.SKILLS.PLACE.type == "modular_debug":
                action = self._modular_place(obs)
                return action, vis_inputs
            else:
                raise NotImplementedError
        return DiscreteNavigationAction.STOP, vis_inputs
