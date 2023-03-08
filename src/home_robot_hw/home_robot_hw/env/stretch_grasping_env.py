from typing import Any, Dict, Optional

import numpy as np

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.env.stretch_abstract_env import StretchEnv

REAL_WORLD_CATEGORIES = [
    "other",
    "cup",
    "other",
]


DETIC = "detic"


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def __init__(self, segmentation_method=DETIC, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES

        self.segmentation_method = segmentation_method
        if self.segmentation_method == DETIC:
            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(self.goal_options),
                sem_gpu_id=0,
            )

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

        if continuous_action is not None:
            if not self.in_navigation_mode():
                self.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            self.navigate_to(continuous_action, relative=True, blocking=True)
        rospy.sleep(0.5)

    def get_observation(self) -> Observations:
        rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=True)

        # Create the observation
        obs = Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=np.zeros(2),  # TODO Replace
            compass=np.zeros(1),  # TODO Replace
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
            },
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
