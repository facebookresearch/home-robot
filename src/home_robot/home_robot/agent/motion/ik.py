# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pybullet as p


class PybulletIKSolver:
    def __init__(self, urdf_path, ee_link_name, controlled_joints=None):
        self.pc_id = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.pc_id,
        )

        self.ee_idx = self.get_link_names().index(ee_link_name)
        self.controlled_joints = np.array(controlled_joints, dtype=np.int32)

    def get_joint_names(self):
        names = []
        for i in range(p.getNumJoints(self.robot_id)):
            names.append(p.getJointInfo(self.robot_id, i)[1].decode("utf-8"))
        return names

    def get_link_names(self):
        names = []
        for i in range(p.getNumJoints(self.robot_id)):
            names.append(p.getJointInfo(self.robot_id, i)[12].decode("utf-8"))
        return names

    def get_num_joints(self):
        return p.getNumJoints(self.robot_id, self.pc_id)

    def compute_ik(self, pos_desired, quat_desired):
        # TODO - remove debug code
        #qq = [p[0] for p in p.getJointStates(self.robot_id,
        #    #jointIndices=np.arange(self.get_num_joints()),
        #    jointIndices=np.arange(self.ee_idx),
        #    physicsClientId=self.pc_id)]

        q_full = np.array(
            p.calculateInverseKinematics(
                self.robot_id,
                self.ee_idx,
                pos_desired,
                quat_desired,
                maxNumIterations=1000,
                residualThreshold=1e-6,
            )
        )

        if self.controlled_joints is not None:
            q_out = q_full[self.controlled_joints]
        else:
            q_out = q_full

        return q_out
