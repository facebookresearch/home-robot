import numpy as np
from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations, DiscreteNavigationAction
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot.perception.detection.detic.detic_perception import DeticPerception


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def __init__(self, *args, **kwargs):
        # TODO Specify confidence threshold as a parameter
        # self.segmentation = DeticPerception(
        #     vocabulary="custom",
        #     custom_vocabulary=",".join(self.goal_options),
        #     sem_gpu_id=0,
        # )

    def reset(self):
        pass

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        pass

    def get_observation(self) -> Observations:
        pass
        # rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=True)
        #
        # # Create the observation
        # obs = Observations(
        #     rgb=rgb.copy(),
        #     depth=depth.copy(),
        #     gps=np.zeros(2),      # TODO Replace
        #     compass=np.zeros(1),  # TODO Replace
        # )
        # # Run the segmentation model here
        # obs = self.segmentation.predict(obs, depth_threshold=0.5)
        # obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1
        # return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
