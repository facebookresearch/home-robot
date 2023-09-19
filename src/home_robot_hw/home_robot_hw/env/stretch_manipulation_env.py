# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import numpy as np
import rospy
import trimesh

from home_robot.core.interfaces import Action
from home_robot.motion.stretch import (
    STRETCH_BASE_FRAME,
    STRETCH_CAMERA_FRAME,
    STRETCH_GRASP_FRAME,
    HelloStretchIdx,
    HelloStretchKinematics,
)
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchManipulationEnv(StretchEnv):
    """Simple environment to enable rolling out manipulation policies via SLAP"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot = HelloStretchKinematics(ik_type="pinocchio")

    def reset(self) -> None:
        """Reset is called at the beginning of each episode where the
        robot retracts gripper and sets itself to a neutral position"""
        # TODO: implement this
        raise NotImplementedError

    def apply_action(self, manip_action: Optional[Dict[str, Any]] = None) -> None:
        """
        manip_action: Manipulation action in cartesian space
                      (pos, quat)
        """
        # TODO: add gripper-action to StretchManipulationEnv.apply_action
        if manip_action is None or "pos" not in manip_action.keys():
            # TODO modify this to generate a dictionary using current pose
            current_pose = self.get_pose(STRETCH_GRASP_FRAME, STRETCH_BASE_FRAME)
            manip_action = {
                "pos": current_pose[0].reshape(-1),
                "ori": current_pose[1].reshape(-1),
                "gripper": manip_action["gripper"]
                if "gripper" in manip_action.keys()
                else 0,
            }
        q0, _ = self.update()
        q, success, ik_debug_info = self.robot.manip_ik(
            (manip_action["pos"], manip_action["rot"]), q0=q0
        )
        self.goto(q, wait=True, move_base=True)
        self._move_gripper(manip_action["gripper"])
        print("Moved to predicted action")

    def _move_gripper(self, gripper: int):
        q, _ = self.update()
        if gripper == 1:
            close_q = self.robot.config_close_gripper(q)
            self.goto(close_q, wait=True, move_base=True)
        else:
            open_q = self.robot.config_open_gripper(q)
            self.goto(open_q, wait=True, move_base=True)

    def get_gripper_state(self, q: np.ndarray):
        """returns gripper state from full joint state"""
        return q[HelloStretchIdx.GRIPPER]

    # over-riding the following methods from the parent class
    def episode_over(self) -> None:
        raise NotImplementedError

    def get_episode_metrics(self) -> None:
        raise NotImplementedError

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Collects sensor data and passes as a dictionary to SLAP
        1. rgb image
        2. depth image
        3. base pose
        4. joint positions
        5. joint velocities
        6. camera pose
        7. camera info
        8. end-effector pose
        """
        # record rgb and depth
        # camera_pose = self.get_camera_pose_matrix(rotated=False)
        camera_pose = self.get_pose(STRETCH_CAMERA_FRAME, "base_link")
        rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=False)
        q, dq = self.update()
        ee_pose_0 = self.robot.fk(q)
        ee_pose = np.concatenate((ee_pose_0[0], ee_pose_0[1]), axis=0)
        gripper_state = q[HelloStretchIdx.GRIPPER]
        # TODO get the ee-pose from TF lookup
        # ee_pose_1 = self.get_pose("link_grasp_center", base_frame=STRETCH_BASE_FRAME)
        base_pose = self.get_base_pose()
        # TODO convert the following to use home_robot.core.interface.Observations
        observations = {
            "rgb": rgb,
            "depth": depth,
            "xyz": xyz,
            "q": q,
            "dq": dq,
            "ee_pose": ee_pose,
            "gripper_state": gripper_state,
            "camera_pose": camera_pose,
            "base_pose": base_pose,
        }
        return observations


if __name__ == "__main__":
    rospy.init_node("test_stretch_manipulation_env")
    robot = StretchManipulationEnv()
    robot._move_gripper(0)
