# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import numpy as np
import pybullet as pb

from home_robot.motion.ik_solver_base import IKSolverBase
from home_robot.motion.robot import RobotModel
from home_robot.utils.bullet import PbArticulatedObject, PbClient


class BulletRobotModel(RobotModel):
    """placeholder"""

    def __init__(
        self,
        name: str = "robot",
        urdf_path: Optional[str] = None,
        visualize: bool = False,
        assets_path: Optional[str] = None,
    ):
        # Load and create planner
        self.backend = PbClient(visualize=visualize)
        if urdf_path is not None:
            # Create object reference
            self.ref = self.backend.add_articulated_object(
                name, urdf_path, assets_path=assets_path
            )

    def get_backend(self):
        """Return model of the robot in bullet - environment for 3d collision checks"""
        return self.backend


class PybulletIKSolver(IKSolverBase):
    """Create a wrapper for solving inverse kinematics using PyBullet"""

    def __init__(
        self,
        urdf_path,
        ee_link_name,
        controlled_joints,
        joint_range=None,
        visualize=False,
    ):
        self.env = PbClient(visualize=visualize, is_simulation=False)
        self.robot = self.env.add_articulated_object("robot", urdf_path)
        self.pc_id = self.env.id
        self.robot_id = self.robot.id
        self.visualize = visualize
        self.range = joint_range

        # Debugging code, not very robust
        if visualize:
            self.debug_block = PbArticulatedObject(
                "red_block", "./assets/red_block.urdf", client=self.env.id
            )

        self.ee_link_name = ee_link_name
        self.ee_idx = self.get_link_names().index(ee_link_name)
        self.controlled_joints = self.robot.controllable_joints_to_indices(
            controlled_joints
        )
        self.controlled_joints = np.array(self.controlled_joints, dtype=np.int32)

    def get_joint_names(self):
        return self.robot.get_joint_names()

    def get_link_names(self):
        return self.robot.get_link_names()

    def get_num_joints(self):
        return self.robot.get_num_joints()

    def get_num_controllable_joints(self):
        return self.robot.get_num_controllable_joints()

    def set_joint_positions(self, q_init):
        q_full = np.zeros(self.get_num_controllable_joints())
        if q_init.shape[0] == len(self.controlled_joints):
            q_full[self.controlled_joints] = q_init
        else:
            q_full[self.controlled_joints] = q_init[self.controlled_joints]
        self.robot.set_joint_positions(q_full)

    def get_dof(self):
        return len(self.controlled_joints)

    def compute_fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.set_joint_positions(q)
        pos, quat = self.robot.get_link_pose(self.ee_link_name)
        return pos, quat

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_attempts: int = 5,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, dict]:
        q_out = None
        success = False

        if q_init is not None:
            # This version assumes that q_init is NOT in the right format yet
            num_attempts = 1

            # Update initial configuration used in bullet for optimization
            self.set_joint_positions(q_init)
            random_initialization = False
        else:
            random_initialization = True

        if self.visualize:
            self.debug_block.set_pose(pos_desired, quat_desired)
            input("--- Press enter to solve ---")

        for _ in range(num_attempts):
            # Randomly initialize before we attempt pybullet inverse kinematics
            if random_initialization:
                rng = self.range[:, 1] - self.range[:, 0]
                min_range = np.copy(self.range[:, 0])
                rng[np.isinf(rng)] = 0
                min_range[np.isinf(min_range)] = 0
                # Initialize in the middle 80% of joint ranges
                q_init = (np.random.random() * rng) + min_range
                self.set_joint_positions(q_init)

            q_full = np.array(
                pb.calculateInverseKinematics(
                    self.robot_id,
                    self.ee_idx,
                    pos_desired,
                    quat_desired,
                    # maxNumIterations=1000,
                    # residualThreshold=1e-6,
                    physicsClientId=self.pc_id,
                )
            )
            # In the ik format - controllable joints only
            self.robot.set_joint_positions(q_full)
            if self.visualize:
                input("--- Solved. Press enter to finish ---")

            if self.controlled_joints is not None:
                q_out = q_full[self.controlled_joints]
                success = True

                if self.range is not None:
                    if not (
                        np.all(q_out > self.range[:, 0])
                        and np.all(q_out < self.range[:, 1])
                    ):
                        if verbose:
                            print("------")
                            print("IK failure:")
                            print(" min =", self.range[:, 0])
                            print("pred =", q_out)
                            print(" max =", self.range[:, 1])
                            print(q_out > self.range[:, 0])
                            print(q_out < self.range[:, 1])
                        success = False
            else:
                q_out = q_full

            if success:
                break

        if verbose:
            print("-------------------")
            print("Success", success)
            print("Result:", q_out)

        debug_info = {"best_q_out": q_out}
        if not success:
            q_out = None

        return q_out, success, debug_info
