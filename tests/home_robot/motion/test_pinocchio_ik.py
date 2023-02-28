import numpy as np
from scipy.spatial.transform import Rotation as R

from home_robot_hw.agent.motion.pinocchio_ik_solver import (
    CEM, CEM_MAX_ITERATIONS, CEM_NUM_SAMPLES, CEM_NUM_TOP, EE_NAME,
    ORI_ERROR_TOL, PIN_CONTROLLED_JOINTS, POS_ERROR_TOL, URDF_PATH,
    PinocchioIKSolver)

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
