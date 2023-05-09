from typing import List, Optional, Tuple

import numpy as np


class SimpleGraspMotionPlanner(object):
    def __init__(self, robot):
        """
        Solve IK
        """
        self.robot = robot

    def plan_to_grasp(
        self, grasp_pose: Tuple[np.ndarray], initial_cfg: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Create offsets for the full trajectory plan to get to the object.
        Then return that plan."""

        # Make sure we can pull out the position and quaternion
        assert len(grasp_pose) == 2
        grasp_pos, grasp_quat = grasp_pose

        # Save initial waypoint to return to
        initial_pt = ("initial", initial_cfg, False)

        # Create a pregrasp point at the top of the robot's arc
        pregrasp_cfg = initial_cfg.copy()
        pregrasp_cfg[1] = 0.95
        pregrasp = ("pregrasp", pregrasp_cfg, False)

        # Try grasp first - find an IK solution for this
        grasp_cfg, success, _ = self.robot.manip_ik(grasp_pos, grasp_quat)
        if success and grasp_cfg is not None:
            grasp_pt = (
                "grasp",
                self.robot.config_to_manip_command(grasp_cfg),
                True,
            )
        else:
            print("-> could not solve for grasp")
            return None

        # Standoff is 8cm over the grasp for now
        standoff_pos = grasp_pos + np.array([0.0, 0.0, 0.08])
        success, standoff_cfg = self.robot.manip_ik(
            standoff_pos,
            grasp_quat,  # initial_cfg=grasp_cfg
        )
        if success and standoff_cfg is not None:
            standoff = (
                "standoff",
                self.robot.config_to_manip_command(standoff_cfg),
                False,
            )
        else:
            print("-> could not solve for standoff")
            return None
        back_cfg = self.robot.config_to_manip_command(standoff_cfg)
        back_cfg[2] = 0.01
        back = ("back", back_cfg, False)

        return [pregrasp, back, standoff, grasp_pt, standoff, back, initial_pt]
