from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations, DiscreteNavigationAction
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchGraspingEnv(StretchEnv):
    """Create a Detic-based grasping environment"""

    def reset(self):
        pass

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        pass

    def get_observation(self) -> Observations:
        pass

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass
