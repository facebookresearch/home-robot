import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p

# HAB_STRETCH_PATH = "../hab_stretch"
HAB_STRETCH_PATH = "../assets/hab_stretch"

# URDF_REL_PATH = "urdf/stretch_dex_wrist_simplified.urdf"
URDF_REL_PATH = "urdf/planner_calibrated.urdf"
# URDF_REL_PATH = "urdf/planner_calibrated_simplified.urdf"

URDF_PATH = os.path.join(HAB_STRETCH_PATH, URDF_REL_PATH)

LOCK_JOINTS = [1, 2]

EE_IDX = 17


class PybulletIKSolver:
    def __init__(self, urdf_path, ee_idx):
        self.pc_id = p.connect(p.DIRECT)

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.pc_id,
        )

        # Get limits
        self.upper_limits = np.zeros(ee_idx)
        self.lower_limits = np.zeros(ee_idx)
        for i in range(ee_idx):
            if i in LOCK_JOINTS:
                self.upper_limits[i] = 0.0
                self.lower_limits[i] = 0.0
            else:
                joint_info = p.getJointInfo(self.robot_id, i)
                self.lower_limits[i] = joint_info[8]
                self.upper_limits[i] = joint_info[9]

    def print_joints(self):
        for i in range(p.getNumJoints(self.robot_id)):
            print(i, p.getJointInfo(self.robot_id, i)[1])

    def print_links(self):
        for i in range(p.getNumJoints(self.robot_id)):
            print(i, p.getJointInfo(self.robot_id, i)[1])

    def compute_ik(self, pos_desired, quat_desired):
        return p.calculateInverseKinematics(
            self.robot_id,
            EE_IDX,
            pos_desired,
            quat_desired,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
        )


if __name__ == "__main__":
    ik_solver = PybulletIKSolver(URDF_PATH, EE_IDX)

    pos_desired = np.array([0.2, 0.3, 0.5])
    quat_desired = R.from_rotvec(np.array([0, 0, 0])).as_quat()

    q_result = ik_solver.compute_ik(pos_desired, quat_desired)

    print(f"Desired EE pose: pos={pos_desired.tolist()}, quat={quat_desired.tolist()}")
    print(f"Joint pos solution: {q_result}")
