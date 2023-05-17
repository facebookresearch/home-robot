from typing import List

import rospy

from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.stretch_pick_and_place_env import DETIC, StretchPickandPlaceEnv


class LanguagePlannerEnv(StretchPickandPlaceEnv):
    def __init__(
        self,
        config,
        goal_options: List[str] = None,
        segmentation_method: str = ...,
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

    def reset(self):
        # clean goal info if in obs
        rospy.sleep(0.5)  # Make sure we have time to get ROS messages
        self.robot.wait()
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        # Switch control mode on the robot to nav
        # Also set the robot's head into "navigation" mode - facing forward
        self.robot.move_to_nav_posture()
