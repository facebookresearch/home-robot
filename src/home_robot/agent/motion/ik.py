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
        self.controlled_joints = controlled_joints

    def get_joint_names(self):
        names = []
        for i in range(p.getNumJoints(self.robot_id)):
            names.append(p.getJointInfo(self.robot_id, i)[1])
        return names

    def get_link_names(self):
        names = []
        for i in range(p.getNumJoints(self.robot_id)):
            names.append(p.getJointInfo(self.robot_id, i)[12])
        return names

    def compute_ik(self, pos_desired, quat_desired):
        q_full = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_idx,
            pos_desired,
            quat_desired,
        )

        if self.controlled_joints is not None:
            q_out = q_full[self.controlled_joints]
        else:
            q_out = q_full

        return q_out
