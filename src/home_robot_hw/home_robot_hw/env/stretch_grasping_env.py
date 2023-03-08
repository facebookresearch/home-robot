from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.motion.stretch import STRETCH_HOME_Q, STRETCH_PREGRASP_Q
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.utils.grasping import GraspingUtility

REAL_WORLD_CATEGORIES = [
    "other",
    "chair",
    "cup",
    "table" "other",
]

DETIC = "detic"


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def __init__(
        self,
        config=None,
        forward_step=0.25,
        rotate_step=30.0,
        segmentation_method=DETIC,
        visualize_planner=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # Create a visualizer
        if config is not None:
            self.visualizer = Visualizer(config)
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

        self.grasping_utility = GraspingUtility(
            self, visualize_planner=visualize_planner
        )

    def reset(self, goal: str):
        self.set_goal(goal)
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.update()
        self.rgb_cam.wait_for_image()
        self.dpt_cam.wait_for_image()
        self._episode_start_pose = xyt2sophus(self.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

    def try_grasping(self, visualize_masks=False, dry_run=False):
        self.grasping_utility.try_grasping(visualize=visualize_masks, dry_run=dry_run)

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        # TODO Determine what form the grasp action should take and move
        #  grasping execution logic here
        if self.visualizer is not None:
            self.visualizer.visualize(**info)
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
        else:
            # Do nothing if "stop"
            # continuous_action = None
            # if not self.in_manipulation_mode():
            #     self.switch_to_manipulation_mode()
            pass

        if continuous_action is not None:
            if not self.in_navigation_mode():
                self.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            self.navigate_to(continuous_action, relative=True, blocking=True)

        # Sleep after sending the navigate command
        rospy.sleep(0.5)

    def set_goal(self, goal):
        """set a goal as a string"""
        assert goal in self.goal_options
        self.current_goal_id = self.goal_options.index(goal)
        self.current_goal_name = goal

    def sample_goal(self):
        """set a random goal"""
        # idx = np.random.randint(len(self.goal_options) - 2) + 1
        idx = 2
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=True)
        current_pose = xyt2sophus(self.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        # pos, vel, frc = self.get_joint_state()

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb,
            depth=depth,
            xyz=xyz,
            gps=gps,
            compass=np.array([theta]),
            # base_pose=sophus2obs(relative_pose),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
            },
            # joint_positions=pos,
        )
        # Run the segmentation model here
        if self.segmentation_method == DETIC:
            obs = self.segmentation.predict(obs)
            obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1
            obs.task_observations["goal_mask"] = obs.semantic == self.current_goal_id
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
