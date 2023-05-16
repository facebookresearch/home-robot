import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import rospy

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
    STRETCH_PREGRASP_Q,
)
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.config import get_config
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import StretchClient
from home_robot_hw.utils.grasping import GraspPlanner

REAL_WORLD_CATEGORIES = [
    "other",
    "chair",
    "cup",
    "table",
    "other",
]  # TODO: Remove hardcoded indices in the visualizer so we can add more objects


DETIC = "detic"


def load_config(visualize=False, print_images=True, **kwargs):
    config_path = "projects/stretch_ovmm/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    config.freeze()
    return config


class StretchPickandPlaceEnv(StretchEnv):
    """Create a Detic-based pick and place environment"""

    # Number of degrees of freedom in our robot joints action space
    joints_dof = 10

    def __init__(
        self,
        config,
        goal_options: List[str] = None,
        segmentation_method: str = DETIC,
        visualize_planner: bool = False,
        ros_grasping: bool = True,
        test_grasping: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        """
        Defines discrete planning environment.

        ros_grasping: create ROS grasp planner
        debug: pause between motions; slows down execution to debug specific behavior
        """
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        if goal_options is None:
            goal_options = REAL_WORLD_CATEGORIES
        self.goal_options = goal_options
        self.forward_step = config.ENVIRONMENT.forward
        self.rotate_step = np.radians(config.ENVIRONMENT.turn_angle)
        self.test_grasping = test_grasping
        self.dry_run = dry_run
        self.debug = debug

        self.robot = StretchClient(init_node=False)

        # Create a visualizer
        if config is not None:
            self.visualizer = Visualizer(config)
            config.defrost()
            config.AGENT.SEMANTIC_MAP.num_sem_categories = len(self.goal_options)
            config.freeze()
        else:
            self.visualizer = None

        # Set up the segmenter
        self.segmentation_method = segmentation_method
        if self.segmentation_method == DETIC:
            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(self.goal_options),
                sem_gpu_id=0,
            )

        if ros_grasping:
            # Create a simple grasp planner object, which will let us pick stuff up.
            # This takes in a reference to the robot client - will replace "self" with "self.client"
            self.grasp_planner = GraspPlanner(
                self.robot, self, visualize_planner=visualize_planner
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

    def reset(self, goal_find: str, goal_obj: str, goal_place: str):
        """Reset the robot and prepare to run a trial. Make sure we have images and up to date state info."""
        self.set_goal(goal_find, goal_obj, goal_place)
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.robot.wait()
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        # Switch control mode on the robot to nav
        # Also set the robot's head into "navigation" mode - facing forward
        self.robot.move_to_nav_posture()

    def get_robot(self):
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
        """Rotate the robot back to face forward"""
        if not self.robot.in_navigation_mode():
            self.robot.switch_to_navigation_mode()
            rospy.sleep(self.msg_delay_t)
        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if not self.dry_run and not self.test_grasping:
            self.robot.nav.navigate_to([0, 0, -np.pi / 2], relative=True, blocking=True)
            self.robot.move_to_nav_posture()

    def _switch_to_manip_mode(self):
        """Rotate the robot and put it in the right configuration for grasping"""
        print("PICK UP THE TARGET OBJECT")
        print(" - Robot in navigation mode:", self.robot.in_navigation_mode())
        if not self.robot.in_navigation_mode():
            self.robot.switch_to_navigation_mode()
            rospy.sleep(self.msg_delay_t)
        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if not self.dry_run and not self.test_grasping:
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True, blocking=True)
            self.robot.move_to_manip_posture()

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        """Handle all sorts of different actions we might be inputting into this class. We provide both a discrete and a continuous action handler."""
        # Process the action so we know what to do with it
        if not isinstance(action, HybridAction):
            action = HybridAction(action)
        # Update the visualizer
        if self.visualizer is not None and info is not None:
            self.visualizer.visualize(**info)
        # By default - no arm control
        joints_action = None
        # Handle discrete actions first
        if action.is_discrete():
            action = action.get()
            continuous_action = np.zeros(3)
            if action == DiscreteNavigationAction.MOVE_FORWARD:
                print("FORWARD")
                continuous_action[0] = self.forward_step
            elif action == DiscreteNavigationAction.TURN_RIGHT:
                print("TURN RIGHT")
                continuous_action[2] = -self.rotate_step
            elif action == DiscreteNavigationAction.TURN_LEFT:
                print("TURN LEFT")
                continuous_action[2] = self.rotate_step
            elif action == DiscreteNavigationAction.STOP:
                print("DONE!")
                # Do nothing if "stop"
                continuous_action = None
                # if not self.robot.in_manipulation_mode():
                #     self.robot.switch_to_manipulation_mode()
                pass
            elif action == DiscreteNavigationAction.EXTEND_ARM:
                """Extend the robot arm"""
                print("EXTENDING ARM")
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
            elif action == DiscreteNavigationAction.PICK_OBJECT:
                continuous_action = None
                # Run in a while loop until we have succeeded
                while not rospy.is_shutdown():
                    if self.dry_run:
                        # Dummy out robot execution code for perception tests
                        break
                    ok = self.grasp_planner.try_grasping(
                        wait_for_input=self.debug, visualize=self.test_grasping
                    )
                    if ok:
                        break
            else:
                print("Action not implemented in pick-and-place environment:", action)
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
        # Handle the joints action
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
            breakpoint()
            # Convert into a position command
            positions, pan, tilt = self.robot.model.hab_to_position_command(joints_goal)
            # Now we can send it to the robot
            print("SENDING JOINT POS", positions, "PAN", pan, "TILT", tilt)
            self.robot.head.set_pan_tilt(pan, tilt)
            self.robot.manip.goto_joint_positions(positions, move_base=False)

    def set_goal(self, goal_find: str, goal_obj: str, goal_place: str):
        """Set the goal class as a string. Goal should be an object class we want to pick up."""
        for goal in [goal_find, goal_obj, goal_place]:
            if goal not in self.goal_options:
                raise RuntimeError(
                    f"Goal not supported: {goal} not in {str(self.goal_options)}"
                )
        goal_obj_id = self.goal_options.index(goal_obj)
        goal_find_id = self.goal_options.index(goal_find)
        goal_place_id = self.goal_options.index(goal_place)
        self.task_info = {
            "object_name": goal_obj,
            "start_recep_name": goal_find,
            "place_recep_name": goal_place,
            "object_id": goal_obj_id,
            "start_recep_id": goal_find_id,
            "place_recep_id": goal_place_id,
            "goal_name": f"{goal_obj} from {goal_find} to {goal_place}",
            # Consistency - add ids for the first task
            "object_goal": goal_obj_id,
            "recep_goal": goal_find_id,
        }
        if self.clip_embeddings is not None:
            # TODO: generate on fly if not available
            self.task_info["object_embedding"] = self.clip_embeddings[goal_obj]

        self.current_goal_id = self.goal_options.index(goal_obj)
        self.current_goal_name = goal

    def sample_goal(self):
        """set a random goal"""
        goal_obj_idx = np.random.randint(len(self.goal_options))
        goal_obj = self.goal_options[goal_obj_idx]
        self.current_goal_id = self.goal_options.index(goal_obj_idx)
        self.current_goal_name = goal_obj

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
            # camera_pose=self.get_camera_pose_matrix(rotated=True),
            camera_pose=self.robot.head.get_pose(rotated=True),
            joint=self.robot.model.config_to_hab(joint_positions),
            relative_resting_position=np.array([0.3878479, 0.12924957, 0.4224413]),
            is_holding=np.array([0.0]),
        )
        # Run the segmentation model here
        if self.segmentation_method == DETIC:
            obs = self.segmentation.predict(obs)

            # Make sure we only have one "other" - for ??? some reason
            obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1

            # Choose instance mask with highest score for goal mask
            instance_scores = obs.task_observations["instance_scores"].copy()
            class_mask = (
                obs.task_observations["instance_classes"] == self.current_goal_id
            )

            # If we detected anything... check to see if our target object was found, and if so pass in the mask.
            if len(instance_scores) and np.any(class_mask):
                chosen_instance_idx = np.argmax(instance_scores * class_mask)
                obs.task_observations["goal_mask"] = (
                    obs.task_observations["instance_map"] == chosen_instance_idx
                )
            else:
                obs.task_observations["goal_mask"] = np.zeros_like(obs.semantic).astype(
                    bool
                )

        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
