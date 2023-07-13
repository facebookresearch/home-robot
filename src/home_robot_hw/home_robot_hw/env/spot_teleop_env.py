from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations
from home_robot_hw.env.spot_abstract_env import SpotEnv


class SpotTeleopEnv(SpotEnv):
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        # TODO
        pass
