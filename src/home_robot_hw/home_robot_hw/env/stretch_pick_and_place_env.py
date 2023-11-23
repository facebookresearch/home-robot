# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import pickle
from enum import IntEnum, auto
from typing import Any, Dict, Optional

import clip
import numpy as np
import rospy
import torch
import trimesh.transformations as tra

import home_robot
from home_robot.core.interfaces import (
    Action,
    DiscreteNavigationAction,
    HybridAction,
    Observations,
)
from home_robot.motion.stretch import (
    STRETCH_ARM_EXTENSION,
    STRETCH_ARM_LIFT,
    STRETCH_HOME_Q,
    STRETCH_POSTNAV_Q,
    STRETCH_PREGRASP_Q,
)
from home_robot.perception.wrapper import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.constants import relative_resting_position
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import StretchClient
from home_robot_hw.utils.grasping import GraspPlanner


class SemanticVocab(IntEnum):
    FULL = auto()
    SIMPLE = auto()
    ALL = auto()


class StretchPickandPlaceEnv(StretchEnv):
    """Create a Detic-based pick and place environment"""

    # Number of degrees of freedom in our robot joints action space
    joints_dof = 10

    def __init__(
        self,
        config,
        cat_map_file: str,
        visualize_planner: bool = False,
        ros_grasping: bool = True,
        test_grasping: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        visualize_grasping: bool = False,
        *args,
        **kwargs,
    ):
        """
        Defines discrete planning environment.

        ros_grasping: create ROS grasp planner
        debug: pause between motions; slows down execution to debug specific behavior
        """
        super().__init__(*args, **kwargs)

        self.forward_step = config.ENVIRONMENT.forward
        self.rotate_step = np.radians(config.ENVIRONMENT.turn_angle)
        self.test_grasping = test_grasping
        self.dry_run = dry_run
        self.debug = debug
        self.visualize_grasping = visualize_grasping
        self.task_info = {}
        self.prev_obs = None
        self.prev_grasp_success = False
        self._gripper_state = False

        with open(cat_map_file) as f:
            self.category_map = json.load(f)

        self.robot = StretchClient(init_node=False)

        # Create a visualizer
        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None

        # Connect to grasp planner via ROS
        if ros_grasping:
            # Create a simple grasp planner object, which will let us pick stuff up.
            # This takes in a reference to the robot client - will replace "self" with "self.client"
            obj_id_to_name = {
                0: config.pick_object,
            }
            simple_rec_id_to_name = {
                0: config.start_recep,
                1: config.goal_recep,
            }

            # Simple vocabulary contains only object and necessary receptacles
            simple_vocab = build_vocab_from_category_map(
                obj_id_to_name, simple_rec_id_to_name
            )
            ovmmper = OvmmPerception(config, 0, True)
            ovmmper.update_vocabulary_list(simple_vocab, SemanticVocab.SIMPLE)
            ovmmper.set_vocabulary(SemanticVocab.SIMPLE)
            self.grasp_planner = GraspPlanner(
                self.robot,
                self,
                visualize_planner=visualize_planner,
                semantic_sensor=ovmmper,
            )

        else:
            if visualize_planner:
                raise RuntimeError(
                    "Param visualize_planner was set to True, but no planner is being created; cannot visualize!"
                )
            self.grasp_planner = None

        self.clip_embeddings = None
        if os.path.exists(config.AGENT.clip_embeddings_file):
            self.clip_embeddings = pickle.load(
                open(config.AGENT.clip_embeddings_file, "rb")
            )
        # Wait for the robot
        self.robot.wait()

    def reset(
        self,
        goal_find: str,
        goal_obj: str,
        goal_place: str,
        set_goal: bool = True,
        open_gripper: bool = True,
    ):
        """Reset the robot and prepare to run a trial. Make sure we have images and up to date
        state info."""
        if set_goal:
            self.set_goal(goal_find, goal_obj, goal_place)
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.robot.wait()
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        if open_gripper:
            # Make sure the gripper is open and ready
            if not self.robot.in_manipulation_mode():
                self.robot.switch_to_manipulation_mode()
            self.robot.manip.open_gripper()

        # Switch control mode on the robot to nav
        # Also set the robot's head into "navigation" mode - facing forward
        self.robot.move_to_nav_posture()
        self.prev_grasp_success = False

    def get_robot(self) -> StretchClient:
        """Return the robot interface."""
        return self.robot

    def execute_joints_action(self, action: np.ndarray):
        """
        Original Arm Action Space: We define the action space that jointly controls (1) arm extension (horizontal), (2) arm height (vertical), (3) gripper wrist’s roll, pitch, and yaw, and (4) the camera’s yaw and pitch. The resulting size of the action space is 10.
        - Arm extension (size: 4): It consists of 4 motors that extend the arm: joint_arm_l0 (index 28 in robot interface), joint_arm_l1 (27), joint_arm_l2 (26), joint_arm_l3 (25)
        - Arm height (size: 1): It consists of 1 motor that moves the arm vertically: joint_lift (23)
        - Gripper wrist (size: 3): It consists of 3 motors that control the roll, pitch, and yaw of the gripper wrist: joint_wrist_yaw (31),  joint_wrist_pitch (39),  joint_wrist_roll (40)
        - Camera (size 2): It consists of 2 motors that control the yaw and pitch of the camera: joint_head_pan (7), joint_head_tilt (8)

        As a result, the original action space is the order of [joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3, joint_lift, joint_wrist_yaw, joint_wrist_pitch, joint_wrist_roll, joint_head_pan, joint_head_tilt] defined in habitat/robots/stretch_robot.py
        """
        assert len(action) == self.joints_dof
        raise NotImplementedError()

    def _switch_to_nav_mode(self):
        """Navigation mode switch"""

        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if not self.dry_run and not self.test_grasping:
            self.robot.move_to_nav_posture()

        """Rotate the robot back to face forward"""
        if not self.robot.in_navigation_mode():
            self.robot.switch_to_navigation_mode()

    def _switch_to_manip_mode(self):
        """Rotate the robot and put it in the right configuration for grasping"""

        # We switch to navigation mode in order to rotate by 90 degrees
        if not self.robot.in_navigation_mode() and not self.dry_run:
            self.robot.switch_to_navigation_mode()

        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if not self.dry_run and not self.test_grasping:
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True, blocking=True)
            self.robot.move_to_manip_posture()

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Handle all sorts of different actions we might be inputting into this class. We provide both a discrete and a continuous action handler."""
        self.prev_obs = prev_obs
        # Process the action so we know what to do with it
        if not isinstance(action, HybridAction):
            action = HybridAction(action)
        # Update the visualizer
        if self.visualizer is not None and info is not None:
            self.visualizer.visualize(**info)
        # By default - no arm control
        joints_action = None
        gripper_action = 0
        # Handle discrete actions first
        if action.is_discrete():
            action = action.get()
            continuous_action = np.zeros(3)
            if action == DiscreteNavigationAction.MOVE_FORWARD:
                print("[ENV] Move forward")
                continuous_action[0] = self.forward_step
            elif action == DiscreteNavigationAction.TURN_RIGHT:
                print("[ENV] TURN RIGHT")
                continuous_action[2] = -self.rotate_step
            elif action == DiscreteNavigationAction.TURN_LEFT:
                print("[ENV] Turn left")
                continuous_action[2] = self.rotate_step
            elif action == DiscreteNavigationAction.STOP:
                # Do nothing if "stop"
                continuous_action = None
                return True
            elif action == DiscreteNavigationAction.EXTEND_ARM:
                """Extend the robot arm"""
                print("[ENV] Extending arm")
                joints_action = self.robot.model.create_action(
                    lift=STRETCH_ARM_LIFT, arm=STRETCH_ARM_EXTENSION
                ).joints
                continuous_action = None
            elif action == DiscreteNavigationAction.MANIPULATION_MODE:
                self._switch_to_manip_mode()
                continuous_action = None
            elif action == DiscreteNavigationAction.NAVIGATION_MODE:
                continuous_action = None
                self._switch_to_nav_mode()
                continuous_action = None
            elif action == DiscreteNavigationAction.POST_NAV_MODE:
                self.robot.move_to_post_nav_posture()
                continuous_action = None
            elif action == DiscreteNavigationAction.PICK_OBJECT:
                print("[ENV] Discrete pick policy")
                continuous_action = None
                # Dummy out robot execution code for perception tests
                if not self.dry_run:
                    ok = self.grasp_planner.try_grasping(
                        wait_for_input=False,
                        visualize=True,  # (self.test_grasping or self.visualize_grasping),
                        max_tries=1,
                    )
                    self.prev_grasp_success = ok
            elif action == DiscreteNavigationAction.SNAP_OBJECT:
                # Close the gripper
                gripper_action = 1
            elif action == DiscreteNavigationAction.DESNAP_OBJECT:
                # Open the gripper
                gripper_action = -1
            else:
                print(
                    "[Env] Action not implemented in pick-and-place environment:",
                    action,
                )
                continuous_action = None
        elif action.is_navigation():
            continuous_action = action.get()
        elif action.is_manipulation():
            joints_action, continuous_action = action.get()

        # Move, if we are not doing anything with the arm
        if continuous_action is not None and not self.test_grasping:
            print("Execute navigation action:", continuous_action)
            if not self.robot.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
                # rospy.sleep(self.msg_delay_t)
            if not self.dry_run:
                print("GOTO", continuous_action)
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=True
                )
        self._handle_joints_action(joints_action)
        self._handle_gripper_action(gripper_action)
        return False

    def _handle_joints_action(self, joints_action: Optional[np.ndarray]):
        """Will convert joints action into the right format and execute it, if it exists."""
        if joints_action is not None:
            # Check to make sure arm control is enabled
            if not self.robot.in_manipulation_mode():
                self.robot.switch_to_manipulation_mode()
            # Now actually move the arm
            # Get current joint positions in habitat coordinates
            current_joint_positions = self.robot.model.config_to_hab(
                self.robot.get_joint_state()[0]
            )
            # Compute position goal from deltas
            joints_goal = joints_action + current_joint_positions
            # Convert into a position command
            positions, pan, tilt = self.robot.model.hab_to_position_command(joints_goal)
            # Now we can send it to the robot
            print("[ENV] SENDING JOINT POS", positions, "PAN", pan, "TILT", tilt)
            self.robot.head.set_pan_tilt(pan, tilt)
            self.robot.manip.goto_joint_positions(positions, move_base=False)
        else:
            # No action to handle
            pass

    def _handle_gripper_action(self, gripper_action: int):
        """Handle any gripper action. Positive = close; negative = open; 0 = do nothing."""
        if gripper_action > 0:
            # Close the gripper
            if not self.robot.in_manipulation_mode():
                self.robot.switch_to_manipulation_mode()
            self.robot.manip.close_gripper()
            self._gripper_state = True
        elif gripper_action < 0:
            # Open the gripper
            if not self.robot.in_manipulation_mode():
                self.robot.switch_to_manipulation_mode()
            self.robot.manip.open_gripper()
            self._gripper_state = False
        else:
            # If the gripper action was zero, do nothing!
            pass

    def set_goal(
        self,
        goal_find: str,
        goal_obj: str,
        goal_place: str,
        check_receptacles=True,
    ):
        """Set the goal class as a string. Goal should be an object class we want to pick up."""
        if check_receptacles:
            recep_name_map = self.category_map["recep_category_to_recep_category_id"]
            for goal in [goal_find, goal_place]:
                if goal not in recep_name_map:
                    raise RuntimeError(
                        f"Receptacle goal not supported: {goal} not in {str(list(recep_name_map.keys()))}"
                    )
        else:
            recep_name_map = None
        self.task_info = {
            "object_name": goal_obj,
            "start_recep_name": goal_find,
            "place_recep_name": goal_place,
            "goal_name": f"{goal_obj} from {goal_find} to {goal_place}",
            "start_receptacle": recep_name_map[goal_find]
            if recep_name_map is not None
            else -1,
            "goal_receptacle": recep_name_map[goal_place]
            if recep_name_map is not None
            else -1,
            # # To be populated by the agent
            "recep_idx": -1,
            "semantic_max_val": -1,
            "object_goal": -1,
            "start_recep_goal": -1,
            "end_recep_goal": -1,
        }

        if self.clip_embeddings is not None and goal_obj in self.clip_embeddings:
            self.task_info["object_embedding"] = self.clip_embeddings[goal_obj]
        else:
            # generate clip embeddings by loading clip model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device)

            # Prepare the inputs
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {goal_obj}")]).to(
                device
            )

            # Get CLIP embeddings
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
            self.task_info["object_embedding"] = text_features[0].cpu().numpy()

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from the robot. Read RGB + depth + point cloud from the robot's cameras, get current pose, and use all of this to compute the observations

        Returns:
            obs: observations containing everything the robot policy will be using to make decisions, other than its own internal state.
        """
        rgb, depth, xyz = self.robot.head.get_images(
            compute_xyz=True,
        )
        current_pose = xyt2sophus(self.robot.nav.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Get joint state information
        joint_positions, _, _ = self.robot.get_joint_state()

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations=self.task_info,
            camera_pose=self.robot.head.get_pose(rotated=True),
            joint=self.robot.model.config_to_hab(joint_positions),
            relative_resting_position=relative_resting_position,
        )
        print("no rwyz", obs.camera_pose[:3, :3])
        roll, pitch, yaw = tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        obs.task_observations["prev_grasp_success"] = np.array(
            [self.prev_grasp_success], np.float32
        )
        obs.task_observations[
            "in_manipulation_mode"
        ] = self.robot.in_manipulation_mode()
        obs.task_observations["in_navigation_mode"] = self.robot.in_navigation_mode()
        obs.task_observations[
            "base_camera_pose"
        ] = self.robot.head.get_pose_in_base_coords(rotated=True)
        obs.task_observations["gripper-state"] = self._gripper_state

        self.prev_obs = obs
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
