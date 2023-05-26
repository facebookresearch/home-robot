from typing import Any, Dict, List, Optional
from std_msgs.msg import Float64
import numpy as np
import rospy
import yaml

import home_robot
from home_robot.core.interfaces import (
    Action,
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    HybridAction,
    Observations,
)
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.stretch_pick_and_place_env import DETIC, StretchPickandPlaceEnv

from home_robot.motion.stretch import (
    STRETCH_HOME_Q,
    STRETCH_NAVIGATION_Q,
    STRETCH_PREGRASP_Q,
)
motion_choices = ["default", "slow", "fast", "very_slow"]


def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_data


def get_velocities(motion_profiles_dict, joint_names, speed: str = "default"):
    assert (
        speed in motion_choices
    ), "Speed must be 'default', 'slow', 'fast', or 'very_slow'"
    joint_velocities = []
    for joint in joint_names:
        joint_velocities.append(motion_profiles_dict[joint][speed]["vel"])
    return joint_velocities


def get_accelerations(motion_profiles_dict, joint_names, speed: str = "default"):
    assert (
        speed in motion_choices
    ), "Speed must be 'default', 'slow', 'fast', or 'very_slow'"
    joint_accelerations = []
    for joint in joint_names:
        joint_accelerations.append(motion_profiles_dict[joint][speed]["accel"])
    return joint_accelerations


class GeneralLanguageEnv(StretchPickandPlaceEnv):
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
        **kwargs
    ):
        super().__init__(
            config,
            goal_options,
            segmentation_method,
            visualize_planner,
            ros_grasping,
            test_grasping,
            dry_run,
            debug,
            *args,
            **kwargs
        )
        self.current_goal_id = None
        self.current_goal_name = None
        self.lin_vel_publisher = rospy.Publisher('goto_controller/v_max', Float64, queue_size=10) 
        self.ang_vel_publisher = rospy.Publisher('goto_controller/w_max', Float64, queue_size=10) 
        self.lin_acc_publisher = rospy.Publisher('goto_controller/acc_lin', Float64, queue_size=10) 
        self.ang_acc_publisher = rospy.Publisher('goto_controller/acc_ang', Float64, queue_size=10) 
        
        root = "src/home_robot/config/control/"
        self.motion_profiles_dict = read_yaml(root + "motion_profiles.yaml")

        self.joints_for_mp = [
            "base",
            "base",
            "base",
            "lift",
            "arm",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
            "stretch_gripper",
            "head_pan",
            "head_tilt",
        ]
        
    def reset(self):
        # TODO (@priyam): clean goal info if in obs
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.robot.wait()
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        # Switch control mode on the robot to nav
        # Also set the robot's head into "navigation" mode - facing forward
        self.robot.move_to_nav_posture()

    def set_goal(self, info: Dict):
        vocab = ["other"] + info["object_list"] + ["other"]
        self.segmentation.reset_vocab(vocab)

        if len(info["object_list"]) > 1:
            self.current_goal_id = 2
            self.current_goal_name = info["object_list"][1]
        else:
            self.current_goal_id = 1
            self.current_goal_name = info["object_list"][0]
        self.speed = info['speed'].lower()
        assert self.speed in motion_choices, "Speed must be 'default', 'slow', 'fast', or 'very_slow'"
            
    def _switch_to_manip_mode(self, grasp_only=True):
        """Rotate the robot and put it in the right configuration for grasping"""

        # We switch to navigation mode in order to rotate by 90 degrees
        if not self.robot.in_navigation_mode() and not self.dry_run:
            self.robot.switch_to_navigation_mode()

        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if grasp_only:
            self.robot.move_to_manip_posture()
            return
        if not self.dry_run and not self.test_grasping:
            self.publish_to_motion_profile_controller()
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True, blocking=True)
            self.robot.move_to_manip_posture()

    def publish_to_motion_profile_controller(self):
        self.lin_vel_publisher.publish(self.motion_profiles_dict['base'][self.speed]["vel"])
        self.lin_acc_publisher.publish(self.motion_profiles_dict['base'][self.speed]["accel"])
        self.ang_vel_publisher.publish(self.motion_profiles_dict['base'][self.speed]["vel"])
        self.ang_acc_publisher.publish(self.motion_profiles_dict['base'][self.speed]["accel"])
        return

    def get_velacc_for_motion_profile(self):
        joint_velocities = get_velocities(self.motion_profiles_dict, self.joints_for_mp, self.speed)
        joint_accelerations = get_accelerations(self.motion_profiles_dict, self.joints_for_mp, self.speed)
        return joint_velocities, joint_accelerations
        
        
    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        """Handle all sorts of different actions we might be inputting into this class.
        We provide both a discrete and a continuous action handler."""
        # Process the action so we know what to do with it
        if not isinstance(action, HybridAction):
            action = HybridAction(action)
        # Update the visualizer
        if self.visualizer is not None and info is not None and "viz" in info:
            self.visualizer.visualize(**info["viz"])
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
                joints_action = self.robot.model. (
                    lift=STRETCH_ARM_LIFT, arm=STRETCH_ARM_EXTENSION
                ).joints
                continuous_action = None
            elif action == DiscreteNavigationAction.MANIPULATION_MODE:
                # set goal based on info dict here
                self.set_goal(info)
                if not self.robot.in_manipulation_mode():
                    self._switch_to_manip_mode()
                continuous_action = None
            elif action == DiscreteNavigationAction.NAVIGATION_MODE:
                # set goal based on info dict here
                self.set_goal(info)
                continuous_action = None
                self._switch_to_nav_mode()
                continuous_action = None
            elif action == DiscreteNavigationAction.PICK_OBJECT:
                print("[ENV] Discrete pick policy")
                continuous_action = None
                # Run in a while loop until we have succeeded
                while not rospy.is_shutdown():
                    if self.dry_run:
                        # Dummy out robot execution code for perception tests
                        break
                    # set velocity for lift, arm and gripper
                    velocities, accelerations = self.get_velacc_for_motion_profile()
                    ok = self.grasp_planner.try_grasping(
                        velocities=velocities, accelerations=accelerations, 
                        wait_for_input=self.debug, visualize=self.test_grasping
                    )
                    if ok:
                        break
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
            if isinstance(action, ContinuousFullBodyAction):
                joints_action, continuous_action = action.get()
            else:
                pos, ori, gripper = action.get()
                continuous_action = None
                print("[ENV] Receiving a ContinuousEndEffectorAction")
                # TODO: convert pos, ori into a ContinuousFullBodyAction here and execute that

        # Move, if we are not doing anything with the arm
        if continuous_action is not None and not self.test_grasping:
            print("Execute navigation action:", continuous_action)
            if not self.robot.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
                # rospy.sleep(self.msg_delay_t)
            if not self.dry_run:
                print("GOTO", continuous_action)
                self.publish_to_motion_profile_controller()
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=True
                )
        self._handle_joints_action(joints_action)
        self._handle_gripper_action(gripper_action)
        return False

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
        if self.current_goal_id is not None:
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
                    obs.task_observations["goal_mask"] = np.zeros_like(
                        obs.semantic
                    ).astype(bool)

        # TODO: remove debug code
        debug_rgb_bgr = False
        if debug_rgb_bgr:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(121)
            plt.imshow(obs.rgb)
            plt.subplot(122)
            plt.imshow(obs.task_observations["semantic_frame"])
            plt.show()
        return obs
