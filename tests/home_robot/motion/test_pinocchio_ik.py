import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from home_robot.motion.pinocchio_ik_solver import (
    CEM_MAX_ITERATIONS,
    POS_ERROR_TOL,
)
from home_robot.motion.pinocchio_ik_solver import PositionIKOptimizer
from home_robot.motion.stretch import (
    STRETCH_GRASP_OFFSET,
    STRETCH_HOME_Q,
    HelloStretch,
)
from home_robot.utils.bullet import PbArticulatedObject
from home_robot.utils.path import REPO_ROOT_PATH
from home_robot.utils.pose import to_matrix, to_pos_quat


def get_ik_solver(debug=False, override=None):
    urdf_abs_path = os.path.join(REPO_ROOT_PATH, "assets/hab_stretch/urdf/")
    return HelloStretch(
        urdf_path=urdf_abs_path,
        visualize=debug,
        ik_type="pinocchio" if override is None else "pybullet",
    )


def get_ik_optimizer(ik_solver):
    return PositionIKOptimizer(
        ik_solver,
        pos_error_tol=0.005,
        ori_error_range=np.array([0.0, 0.0, 0.2]),  # solve for yaw only
    )


def compute_err(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def quaternion_distance(quat1, quat2):
    return 1 - ((quat1 * quat2).sum() ** 2)


def ik_helper(robot, pos, quat, indicator_block, err_threshold, debug=False):
    """ik test helper function."""
    print("GOAL:", pos, quat)
    indicator_block.set_pose(pos, quat)
    res = robot.manip_ik((pos, quat), STRETCH_HOME_Q, relative=True)
    robot.set_config(res)
    pos2, quat2 = robot.get_ee_pose()
    print("RESULT:", pos2, quat2)
    print("x motion:", res[0])
    err = compute_err(pos2, pos)
    print("error was:", err)
    assert err < err_threshold
    assert quaternion_distance(quat, quat2) < err_threshold
    if debug:
        input("press enter to continue")
    return pos2, quat2, res


def test_pinocchio_ik_optimization():
    pos_desired = np.array([-0.10281811, -0.7189281, 0.71703106])
    # pos_desired = np.array([-0.11556295, -0.51387864,  0.8205258 ])
    quat_desired = np.array([-0.7079143, 0.12421559, 0.1409881, -0.68084526])

    robot = get_ik_solver()
    ik_opt = get_ik_optimizer(robot.manip_ik_solver)

    # Directly solve with IK
    q, _ = robot.manip_ik_solver.compute_ik(pos_desired, quat_desired)
    pos_out1, quat_out1 = robot.manip_ik_solver.compute_fk(q)
    pos_err1 = np.linalg.norm(pos_out1 - pos_desired)

    # Solve with CEM
    q_result, best_cost, last_iter, opt_sigma = ik_opt.compute_ik_opt(
        (pos_desired, quat_desired)
    )
    pos_out2, quat_out2 = robot.manip_ik_solver.compute_fk(q_result)
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
    assert pos_err2 < pos_err1
    assert (
        best_cost <= POS_ERROR_TOL
        or np.all(opt_sigma) <= POS_ERROR_TOL
        or last_iter >= CEM_MAX_ITERATIONS
    )


def ik_test_helper(robot, debug=False, err_threshold=1e-4):
    """
    Goal pos and rot: (array([-0.10281811, -0.7189281 ,  0.71703106], dtype=float32), array([-0.7079143 ,  0.12421559,  0.1409881 , -0.68084526]))
    Current best solution: (array([-0.1350856 , -0.71864623,  0.71646219]), array([ 0.7084716 , -0.12145648, -0.13812223,  0.68135047]))

    2nd Goal pos and rot: (array([-0.01556295, -0.51387864,  0.8205258 ], dtype=float32), array([-0.7090214 ,  0.12297839,  0.14050716, -0.6800168 ]))
    Current best solution: (array([-0.12925884, -0.51288551,  0.8185215 ]), array([ 0.71091503, -0.1131743 , -0.13030495,  0.68177122]))
    """
    block = PbArticulatedObject(
        "red_block",
        os.path.join(REPO_ROOT_PATH, "assets/red_block.urdf"),
        client=robot.ref.client,
    )
    robot.set_config(STRETCH_HOME_Q)
    test_poses = [
        (
            [-0.10281811, -0.7189281, 0.71703106],
            [-0.7079143, 0.12421559, 0.1409881, -0.68084526],
        ),
        (
            [-0.01556295, -0.51387864, 0.8205258],
            [-0.7090214, 0.12297839, 0.14050716, -0.6800168],
        ),
    ]
    test_poses = [
        to_pos_quat(to_matrix(pos, quat) @ STRETCH_GRASP_OFFSET)
        for pos, quat in test_poses
    ]
    test_poses = [robot.get_ee_pose()] + test_poses
    for pos, quat in test_poses:
        print("-------- 1: Inverse kinematics ---------")
        ik_helper(robot, pos, quat, block, err_threshold, debug)

        print("-------- 2: FK + IK Consistency  ---------")
        pos1, quat1 = robot.get_ee_pose()
        ik_helper(robot, pos1, quat1, block, err_threshold, debug)


def test_pinocchio_against_pybullet(debug=False, err_threshold=1e-4):
    pinocchio_robot = get_ik_solver()
    pb_robot = get_ik_solver(override="pybullet")
    pb_block = PbArticulatedObject(
        "red_block",
        os.path.join(REPO_ROOT_PATH, "assets/red_block.urdf"),
        client=pb_robot.ref.client,
    )
    pin_block = PbArticulatedObject(
        "red_block",
        os.path.join(REPO_ROOT_PATH, "assets/red_block.urdf"),
        client=pinocchio_robot.ref.client,
    )
    pinocchio_robot.set_config(STRETCH_HOME_Q)
    pb_robot.set_config(STRETCH_HOME_Q)
    test_poses = [
        (
            [-0.10281811, -0.7189281, 0.71703106],
            [-0.7079143, 0.12421559, 0.1409881, -0.68084526],
        ),
        (
            [-0.01556295, -0.51387864, 0.8205258],
            [-0.7090214, 0.12297839, 0.14050716, -0.6800168],
        ),
    ]
    test_poses = [
        to_pos_quat(to_matrix(pos, quat) @ STRETCH_GRASP_OFFSET)
        for pos, quat in test_poses
    ]
    test_poses = [pb_robot.get_ee_pose()] + test_poses
    for pos, quat in test_poses:
        print("-------- 1: Inverse kinematics ---------")
        pin_pos, pin_quat, pin_q = ik_helper(
            pinocchio_robot, pos, quat, pin_block, err_threshold, debug
        )
        pb_pos, pb_quat, pb_q = ik_helper(
            pb_robot, pos, quat, pb_block, err_threshold, debug
        )
        print(f"Pinocchio: {pin_pos}, {pin_quat}, {pin_q}")
        print(f"PyBullet: {pb_pos}, {pb_quat}, {pb_q}")
        pos_err = compute_err(pin_pos, pb_pos)
        quat_err = quaternion_distance(pin_quat, pb_quat)
        assert pos_err < err_threshold
        assert quat_err < err_threshold

        print("-------- 2: FK + IK Consistency  ---------")
        pos1, quat1 = pinocchio_robot.get_ee_pose()
        pin_pos, pin_quat, pin_q = ik_helper(
            pinocchio_robot, pos1, quat1, pin_block, err_threshold, debug
        )
        pos1, quat1 = pb_robot.get_ee_pose()
        pb_pos, pb_quat, pb_q = ik_helper(
            pb_robot, pos1, quat1, pb_block, err_threshold, debug
        )
        pos_err = compute_err(pin_pos, pb_pos)
        quat_err = quaternion_distance(pin_quat, pb_quat)
        assert pos_err < err_threshold
        assert quat_err < err_threshold


def test_pinocchio_base_ik():
    robot = get_ik_solver()
    ik_test_helper(robot)


if __name__ == "__main__":
    test_pinocchio_against_pybullet()
