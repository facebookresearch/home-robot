import numpy as np
from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations, DiscreteNavigationAction
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot.perception.detection.detic.detic_perception import DeticPerception


REAL_WORLD_CATEGORIES = [
    "other",
    "cup",
    "other",
]


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES

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
        pass

    def get_observation(self) -> Observations:
        rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=True)

        # Create the observation
        obs = Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=np.zeros(2),      # TODO Replace
            compass=np.zeros(1),  # TODO Replace
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
            },
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs)
        obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1
        obs.task_observations["goal_mask"] = obs.semantic == self.current_goal_id
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
