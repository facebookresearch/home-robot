from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import (
    Action,
    DiscreteNavigationAction,
    HybridAction,
    Observations,
)
from home_robot.motion.stretch import STRETCH_HOME_Q, STRETCH_PREGRASP_Q
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import StretchClient
from home_robot_hw.utils.grasping import GraspPlanner

REAL_WORLD_CATEGORIES = [
    "other",
    # Seen object categories
    "bowl",
    "cup",
    "bin",
    # Unseen object categories
    "bucket",
    "elephant",
    "computer_mouse",
    "car_keys",
    # Receptacles
    "chair",
    "table",
    "other",
]
REAL_WORLD_CATEGORIES = [
    "other",
    "chair",
    "cup",
    "table",
    "other",
]


DETIC = "detic"


class StretchPickandPlaceEnv(StretchEnv):
    """Create a Detic-based pick and place environment"""

    def __init__(
        self,
        config=None,
        forward_step=0.25,
        rotate_step=30.0,
        segmentation_method=DETIC,
        visualize_planner=False,
        ros_grasping=True,
        test_grasping=False,
        dry_run=False,
        *args,
        **kwargs,
    ):
        """
        Defines discrete planning environment.

        ros_grasping: create ROS grasp planner
        """
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)
        self.test_grasping = test_grasping
        self.dry_run = dry_run

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

    def reset(self, goal_find: str, goal_obj: str, goal_place: str):
        """Reset the robot and prepare to run a trial. Make sure we have images and up to date state info."""
        self.set_goal(goal_find, goal_obj, goal_place)
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.update()
        self.rgb_cam.wait_for_image()
        self.dpt_cam.wait_for_image()
        self._episode_start_pose = xyt2sophus(self.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        # Switch control mode on the robot to nav
        self.robot.switch_to_navigation_mode()
        if self.grasp_planner is not None:
            # Set the robot's head into "navigation" mode - facing forward
            self.grasp_planner.go_to_nav_mode()

    def try_grasping(self, visualize_masks=False, dry_run=False):
        return self.grasp_planner.try_grasping(
            visualize=visualize_masks, dry_run=dry_run
        )

    def get_robot(self):
        """Return the robot interface."""
        return self.robot

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        """Handle all sorts of different actions we might be inputting into this class. We provide both a discrete and a continuous action handler."""
        action = HybridAction(action)
        if self.visualizer is not None and info is not None:
            self.visualizer.visualize(**info)
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
            elif action == DiscreteNavigationAction.MANIPULATION_MODE:
                print("PICK UP THE TARGET OBJECT")
                print(" - Robot in navigation mode:", self.in_navigation_mode())
                continuous_action = None
                if self.in_navigation_mode():
                    self.switch_to_navigation_mode()
                    rospy.sleep(self.msg_delay_t)
                # Dummy out robot execution code for perception tests
                if not self.dry_run:
                    self.robot.nav.navigate_to(
                        [0, 0, np.pi / 2], relative=True, blocking=True
                    )
                    self.grasp_planner.go_to_manip_mode()
            elif action == DiscreteNavigationAction.PICK_OBJECT:
                continuous_action = None
                while not rospy.is_shutdown():
                    if self.dry_run:
                        # Dummy out robot execution code for perception tests\
                        break
                    ok = self.grasp_planner.try_grasping()
                    if ok:
                        break
            else:
                print("Action not implemented in pick-and-place environment:", action)
                continuous_action = None
        elif action.is_navigation():
            continuous_action = action.get()

        # Move, if we are not doing anything with the arm
        if continuous_action is not None and not self.test_grasping:
            print("Execute navigation action:", continuous_action)
            if not self.robot.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            if not self.dry_run:
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=False
                )

                v_stop = 0.05
                dt = 0.1
                pose_current = self.robot.nav.get_base_pose()
                while True:
                    rospy.sleep(dt)
                    pose_new = self.robot.nav.get_base_pose()
                    v_avg = np.linalg.norm(pose_new - pose_current) / dt
                    if v_avg < v_stop:
                        break
                    pose_current = pose_new

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
        self.current_goal_id = self.goal_options.index(goal_obj)
        self.current_goal_name = goal

    def sample_goal(self):
        """set a random goal"""
        goal_obj_idx = np.random.randint(len(self.goal_options))
        goal_obj = self.goal_options[goal_obj_idx]
        self.current_goal_id = self.goal_options.index(goal_obj_idx)
        self.current_goal_name = goal_obj

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth, xyz = self.robot.head.get_images(
            compute_xyz=True,
        )
        current_pose = xyt2sophus(self.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=gps,
            compass=np.array([theta]),
            # base_pose=sophus2obs(relative_pose),
            task_observations=self.task_info,
            camera_pose=self.get_camera_pose_matrix(rotated=True),
            # joint_positions=pos,
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

            obs.task_observations["goal_mask"] = np.zeros_like(obs.semantic).astype(
                bool
            )
            if len(instance_scores):
                masked_instance_scores = instance_scores * class_mask  # set all non-correct object scores to 0
                if np.max(masked_instance_scores):
                    chosen_instance_idx = np.argmax(masked_instance_scores)
                    obs.task_observations["goal_mask"] = (
                        obs.task_observations["instance_map"] == chosen_instance_idx
                    )

        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
