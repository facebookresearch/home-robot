from typing import Any, Dict, Optional

import numpy as np

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.env.stretch_abstract_env import StretchEnv

# REAL_WORLD_CATEGORIES = ["other", "chair", "mug", "other",]
# REAL_WORLD_CATEGORIES = ["other", "backpack", "other",]
REAL_WORLD_CATEGORIES = [
    "other",
    "chair",
    "mug",
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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        self.segmentation_method = segmentation_method
        if self.segmentation_method == DETIC:
            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(self.goal_options),
                sem_gpu_id=0,
            )

        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None
        self.reset()

    def reset(self):
        # TODO Make this better
        self.current_goal_id = 1
        self.current_goal_name = self.goal_options[self.current_goal_id]

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
        if goal in self.goal_options:
            self.current_goal_id = self.goal_options.index(goal)
            self.current_goal_name = goal
            return True
        else:
            return False

    def sample_goal(self):
        """set a random goal"""
        # idx = np.random.randint(len(self.goal_options) - 2) + 1
        idx = 2
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
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
            rgb=rgb.copy(),
            depth=depth.copy(),
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
