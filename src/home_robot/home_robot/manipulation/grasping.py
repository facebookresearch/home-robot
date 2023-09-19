# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import numpy as np

from home_robot.motion.stretch import HelloStretchIdx, HelloStretchKinematics


class SimpleGraspMotionPlanner(object):
    """Simple top-down grasp motion planner for the Stretch."""

    def __init__(self, robot: HelloStretchKinematics, pregrasp_height=1.2):
        """
        Solve IK
        """
        if not isinstance(robot, HelloStretchKinematics):
            raise RuntimeError(
                "The SimpleGraspMotionPlanner was designed only for Stretch."
            )
        self.robot = robot
        self.pregrasp_height = pregrasp_height

    def plan_to_grasp(
        self,
        grasp_pose: Tuple[np.ndarray],
        initial_cfg: np.ndarray,
        z_standoff=0.4,
    ) -> Optional[List[Tuple]]:
        """Create offsets for the full trajectory plan to get to the object.
        Then return that plan.

        This assumes that the grasp pose is expressed in BASE COORDINATES."""

        # Make sure we can pull out the position and quaternion
        assert len(grasp_pose) == 2
        grasp_pos, grasp_quat = grasp_pose

        # Save initial waypoint to return to
        # TODO: remove this if we decide we do not need it
        # initial_pt = ("initial", initial_cfg, False)

        # Create a pregrasp point at the top of the robot's arc
        pregrasp_cfg = initial_cfg.copy()
        pregrasp_cfg[1] = self.pregrasp_height
        pregrasp = ("pregrasp", pregrasp_cfg, False)

        # Try grasp first - find an IK solution for this
        grasp_cfg, success, _ = self.robot.manip_ik((grasp_pos, grasp_quat), q0=None)
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
        # Overwrite standoff pos.z with a really high value so it comes in from above
        standoff_pos = grasp_pos + np.array([0.0, 0.0, z_standoff])
        standoff_pos[2] = np.min([self.robot.max_arm_height, standoff_pos[2]])
        standoff_cfg, success, _ = self.robot.manip_ik(
            (standoff_pos, grasp_quat), q0=None
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

        # Go back to the top
        back_top_cfg = pregrasp_cfg.copy()
        back_top_cfg[0] = standoff_cfg[0]
        back_top = ("back_top", back_top_cfg, False)

        back_cfg = standoff_cfg.copy()
        back_cfg[HelloStretchIdx.ARM] = 0.01
        back_cfg = self.robot.config_to_manip_command(back_cfg)
        back = ("back", back_cfg, False)

        # Return the full motion plan
        return [
            pregrasp,
            back,
            standoff,
            grasp_pt,
            standoff,
            back_top,
            back,
        ]
