import os

import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R

# Error tolerances
POS_ERROR_TOL = 0.005
ORI_ERROR_TOL = [0.1, 0.1, np.pi / 2]

# CEM
CEM_MAX_ITERATIONS = 5
CEM_NUM_SAMPLES = 50
CEM_NUM_TOP = 10

# URDF
HAB_STRETCH_PATH = "../hab_stretch"
URDF_REL_PATH = "urdf/planner_calibrated_simplified.urdf"
URDF_PATH = os.path.join(HAB_STRETCH_PATH, URDF_REL_PATH)

EE_NAME = "link_straight_gripper"
PIN_CONTROLLED_JOINTS = [
    # "base_x_joint",
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]


class PinocchioIKSolver:
    EPS = 1e-4
    DT = 1e-1
    DAMP = 1e-12

    def __init__(self, urdf_path, ee_link_name, controlled_joints):
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pinocchio.neutral(self.model)

        self.ee_frame_idx = [f.name for f in self.model.frames].index(ee_link_name)
        self.controlled_joints = [
            self.model.idx_qs[self.model.getJointId(j)] if j != "ignore" else -1
            for j in controlled_joints
        ]

    def get_dof(self) -> int:
        """returns dof for the manipulation chain"""
        return len(self.controlled_joints)

    def _qmap_control2model(self, q_input):
        q_out = self.q_neutral.copy()
        for i, joint_idx in enumerate(self.controlled_joints):
            q_out[joint_idx] = q_input[i]

        return q_out

    def _qmap_model2control(self, q_input):
        q_out = np.empty(len(self.controlled_joints))
        for i, joint_idx in enumerate(self.controlled_joints):
            if joint_idx >= 0:
                q_out[i] = q_input[joint_idx]

        return q_out

    def compute_fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """given joint values return end-effector position and quaternion associated with it"""
        q_model = self._qmap_control2model(q)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        pinocchio.updateFramePlacement(self.model, self.data, self.ee_frame_idx)
        pos = self.data.oMf[self.ee_frame_idx].translation
        quat = R.from_matrix(self.data.oMf[self.ee_frame_idx].rotation).as_quat()

        return pos.copy(), quat.copy()

    def compute_ik(self, pos, quat, max_iterations=100) -> tuple[np.ndarray, bool]:
        """given end-effector position and quaternion, return joint values"""
        i = 0
        q = self.q_neutral.copy()
        desired_ee_pose = pinocchio.SE3(R.from_quat(quat).as_matrix(), pos)
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacement(self.model, self.data, self.ee_frame_idx)

            dMi = desired_ee_pose.actInv(self.data.oMf[self.ee_frame_idx])
            err = pinocchio.log(dMi).vector
            if np.linalg.norm(err) < self.EPS:
                success = True
                break
            if i >= max_iterations:
                success = False
                break
            J = pinocchio.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_idx,
                pinocchio.ReferenceFrame.LOCAL,
            )
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.DAMP * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            i += 1

        q_control = self._qmap_model2control(q.flatten())

        return q_control, success

    def compute_ik_opt(self, pose_query):
        max_iterations = 30
        num_samples = 100
        num_top = 10  # TODO: what is this?
        pos_error_tol = 0.005
        ori_error_tol = 0.2
        pos_desired, quat_desired = pose_query
        ik_solver = self

        opt = CEM(
            max_iterations=max_iterations,
            num_samples=num_samples,
            num_top=num_top,
            tol=pos_error_tol,
        )

        def solve_ik(dr):
            pos = pos_desired
            quat = (R.from_rotvec(dr) * R.from_quat(quat_desired)).as_quat()

            q, ik_success = ik_solver.compute_ik(pos, quat)
            pos_out, rot_out = ik_solver.compute_fk(q)

            cost_pos = np.linalg.norm(pos - pos_out)
            cost_rot = (
                1 - (rot_out * quat_desired).sum() ** 2
            )  # TODO: just minimize dr?

            cost = cost_pos  # + cost_rot  # TODO: scaling?

            return cost, q

        cost_opt, q_result = opt.optimize(
            solve_ik, x0=np.zeros(3), sigma0=np.array([0, 0, ori_error_tol / 2])
        )
        pos_out, quat_out = ik_solver.compute_fk(q_result)
        print(
            f"After ik optimization, cost: {cost_opt}, result: {pos_out, quat_out} vs desired: {pose_query}"
        )
        # pos_out, quat_out = self.fk(q_result)
        return q_result


class CEM:
    def __init__(self, max_iterations, num_samples, num_top, tol):
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.num_top = num_top
        self.cost_tol = tol

    def optimize(self, func, x0, sigma0):
        i = 0
        mu = x0
        sigma = sigma0
        while True:
            # Sample x
            x_arr = mu + sigma * np.random.randn(self.num_samples, x0.shape[0])

            # Compute costs
            cost_arr = np.zeros(self.num_samples)
            aux_outputs = [None for _ in range(self.num_samples)]
            for j, x in enumerate(x_arr):
                cost_arr[j], aux_outputs[j] = func(x)

            # Sort costs
            idx_sorted_arr = np.argsort(cost_arr)
            i_best = idx_sorted_arr[0]

            # Check termination
            i += 1
            if (
                i >= self.max_iterations
                or cost_arr[i_best] <= self.cost_tol
                or np.all(sigma <= self.cost_tol / 10)
            ):  # TODO: sigma thresh?
                break

            # Update distribution
            mu = np.mean(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)
            sigma = np.std(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)

        return cost_arr[i_best], aux_outputs[i_best]


if __name__ == "__main__":
    pos_desired = np.array([-0.10281811, -0.7189281, 0.71703106])
    # pos_desired = np.array([-0.11556295, -0.51387864,  0.8205258 ])
    quat_desired = np.array([-0.7079143, 0.12421559, 0.1409881, -0.68084526])

    ik_solver = PinocchioIKSolver(URDF_PATH, EE_NAME, PIN_CONTROLLED_JOINTS)

    # Directly solve with IK
    q, _ = ik_solver.compute_ik(pos_desired, quat_desired)
    pos_out1, quat_out1 = ik_solver.compute_fk(q)
    pos_err1 = np.linalg.norm(pos_out1 - pos_desired)

    # Solve with CEM
    opt = CEM(
        max_iterations=CEM_MAX_ITERATIONS,
        num_samples=CEM_NUM_SAMPLES,
        num_top=CEM_NUM_TOP,
        tol=POS_ERROR_TOL,
    )

    def solve_ik(dr):
        pos = pos_desired
        quat = (R.from_rotvec(dr) * R.from_quat(quat_desired)).as_quat()

        q, _ = ik_solver.compute_ik(pos, quat)
        pos_out, _ = ik_solver.compute_fk(q)

        cost = np.linalg.norm(pos - pos_out)

        return cost, q

    _, q_result = opt.optimize(
        solve_ik, x0=np.zeros(3), sigma0=np.array(ORI_ERROR_TOL) / 2
    )
    pos_out2, quat_out2 = ik_solver.compute_fk(q_result)
    pos_err2 = np.linalg.norm(pos_out2 - pos_desired)

    print(f"Desired EE pose: pos={pos_desired.tolist()}, quat={quat_desired.tolist()}")
    print("---------Without CEM---------")
    print(
        f"Resulting EE pose via FK: pos={pos_out1.tolist()}, quat={quat_out1.tolist()}"
    )
    print(f"Pos error: {pos_err1}")
    print("---------With CEM---------")
    print(
        f"Resulting EE pose via FK: pos={pos_out2.tolist()}, quat={quat_out2.tolist()}"
    )
    print(f"Pos error: {pos_err2}")
