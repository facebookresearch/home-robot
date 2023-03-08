from typing import Any, Dict, Optional

import numpy as np
import rospy

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.utils.grasping import GraspingUtility

REAL_WORLD_CATEGORIES = [
    "other",
    "cup",
    "other",
]

DETIC = "detic"


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def __init__(
        self, segmentation_method=DETIC, visualize_planner=False, *args, **kwargs
    ):
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

        self.grasping_utility = GraspingUtility(
            self, visualize_planner=visualize_planner
        )

    def reset(self, goal_category: str):
        assert goal_category in REAL_WORLD_CATEGORIES
        self.current_goal_id = REAL_WORLD_CATEGORIES.index(goal_category)
        self.current_goal_name = goal_category
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.update()
        self.rgb_cam.wait_for_image()
        self.dpt_cam.wait_for_image()

    def try_grasping(self, visualize_masks=False, dry_run=False):
        self.grasping_utility.try_grasping(visualize=visualize_masks, dry_run=dry_run)

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        # TODO apply_action() should call try_grasping()
        pass

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
