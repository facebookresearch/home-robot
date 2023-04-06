# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from home_robot.motion.pinocchio_ik_solver import PositionIKOptimizer
from home_robot.motion.stretch import (
    STRETCH_GRASP_OFFSET,
    STRETCH_HOME_Q,
    HelloStretchKinematics,
)
from home_robot.utils.bullet import PbArticulatedObject
from home_robot.utils.path import REPO_ROOT_PATH
from home_robot.utils.pose import to_matrix, to_pos_quat

# Hyperparams
DEBUG = False
URDF_ABS_PATH = os.path.join(REPO_ROOT_PATH, "assets/hab_stretch/urdf/")

POS_ERROR_TOL = 1.5e-4  # 0.1 mm
ORI_ERROR_TOL = 1e-6  # 0.1 degrees in quat distance

CEM_POS_ERROR_TOL = 0.005
CEM_YAW_ERROR_TOL = 0.2

# Test data (pos, quat)
TEST_DATA = [
    (
        [-0.10281811, -0.7189281, 0.71703106],
        [-0.7079143, 0.12421559, 0.1409881, -0.68084526],
    ),
    (
        [-0.01556295, -0.51387864, 0.8205258],
        [-0.7090214, 0.12297839, 0.14050716, -0.6800168],
    ),
]

TEST_JOINTS = [
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 3, 1, 1, 1, 1, 8, 7, 6],
    )
]


# Helper functions


def compute_err(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def quaternion_distance(quat1, quat2):
    """
    Returns: (1 - cos(theta_diff)) / 2
    """
    return 1 - ((quat1 * quat2).sum() ** 2)


def ik_helper(
    robot, pos, quat, indicator_block=None, debug=DEBUG, ori_error_tol=ORI_ERROR_TOL
):
    """ik test helper function."""
    print("GOAL:", pos, quat)
    if indicator_block is not None:
        indicator_block.set_pose(pos, quat)
    res, success, ik_debug_info = robot.manip_ik(
        (pos, quat), np.zeros(robot.dof), relative=True, verbose=True
    )
    if res is not None:
        robot.set_config(res)
    pos2, quat2 = robot.get_ee_pose()
    print("RESULT:", pos2, quat2)
    print("x motion:", res[0])
    err = compute_err(pos2, pos)
    print("error was:", err)
    assert err < POS_ERROR_TOL
    assert quaternion_distance(quat, quat2) < ori_error_tol
    assert success
    if debug:
        input("press enter to continue")
    return pos2, quat2, res


# Tests


@pytest.fixture(params=TEST_DATA)
def test_pose(request):
    pos_raw, quat_raw = request.param
    pos, quat = to_pos_quat(to_matrix(pos_raw, quat_raw) @ STRETCH_GRASP_OFFSET)
    return pos, quat


@pytest.fixture(params=TEST_JOINTS)
def test_joints(request):
    ros_pose, pin_pose_grnd = request.param
    return ros_pose, pin_pose_grnd


@pytest.fixture
def pb_robot():
    return HelloStretchKinematics(
        urdf_path=URDF_ABS_PATH,
        visualize=DEBUG,
        ik_type="pybullet",
    )


@pytest.fixture
def pin_robot():
    return HelloStretchKinematics(
        urdf_path=URDF_ABS_PATH,
        visualize=DEBUG,
        ik_type="pinocchio",
    )


@pytest.fixture
def pin_optimize_robot():
    return HelloStretchKinematics(
        urdf_path=URDF_ABS_PATH,
        visualize=DEBUG,
        ik_type="pinocchio_optimize",
    )


@pytest.fixture
def pb_optimize_robot():
    return HelloStretchKinematics(
        urdf_path=URDF_ABS_PATH,
        visualize=DEBUG,
        ik_type="pybullet_optimize",
    )


@pytest.fixture
def pb_ik_optimizer(pb_robot):
    return PositionIKOptimizer(
        pb_robot.manip_ik_solver,
        pos_error_tol=CEM_POS_ERROR_TOL,
        ori_error_range=np.array([0.0, 0.0, CEM_YAW_ERROR_TOL]),  # solve for yaw only
    )


@pytest.fixture
def pin_ik_optimizer(pin_robot):
    return PositionIKOptimizer(
        pin_robot.manip_ik_solver,
        pos_error_tol=CEM_POS_ERROR_TOL,
        ori_error_range=np.array([0.0, 0.0, CEM_YAW_ERROR_TOL]),  # solve for yaw only
    )


@pytest.fixture(
    params=["pybullet", "pinocchio", "pybullet_optimize", "pinocchio_optimize"]
)
def robot(request, pb_robot, pin_robot, pb_optimize_robot, pin_optimize_robot):
    if request.param == "pybullet":
        return pb_robot
    elif request.param == "pinocchio":
        return pin_robot
    elif request.param == "pybullet_optimize":
        return pb_optimize_robot
    elif request.param == "pinocchio_optimize":
        return pin_optimize_robot


def test_ik_solvers(robot, test_pose):
    # Loosen the orientation error bounds if we're in an optimize mode
    ori_error_tol = ORI_ERROR_TOL
    if "optimize" in robot._ik_type:
        ori_error_tol = 1e-1

    # Create block for visualization
    block = PbArticulatedObject(
        "red_block",
        os.path.join(REPO_ROOT_PATH, "assets/red_block.urdf"),
        client=robot.ref.client,
    )

    # Set state to home pose
    robot.set_config(STRETCH_HOME_Q)

    # Run ik
    pos, quat = test_pose

    print("-------- 1: Inverse kinematics ---------")
    ik_helper(robot, pos, quat, block, ori_error_tol=ori_error_tol)

    print("-------- 2: FK + IK Consistency  ---------")
    pos1, quat1 = robot.get_ee_pose()
    ik_helper(robot, pos1, quat1, block, ori_error_tol=ori_error_tol)


def test_pinocchio_against_pybullet(pin_robot, pb_robot, test_pose):
    pin_robot.set_config(STRETCH_HOME_Q)
    pb_robot.set_config(STRETCH_HOME_Q)

    # Set state to home pose
    pos, quat = test_pose

    # Run ik
    print("-------- 1: Inverse kinematics ---------")
    pin_pos, pin_quat, pin_q = ik_helper(pin_robot, pos, quat)
    pb_pos, pb_quat, pb_q = ik_helper(pb_robot, pos, quat)
    print(f"Pinocchio: {pin_pos}, {pin_quat}, {pin_q}")
    print(f"PyBullet: {pb_pos}, {pb_quat}, {pb_q}")
    pos_err = compute_err(pin_pos, pb_pos)
    quat_err = quaternion_distance(pin_quat, pb_quat)
    assert pos_err < POS_ERROR_TOL
    assert quat_err < ORI_ERROR_TOL

    print("-------- 2: FK + IK Consistency  ---------")
    pos1, quat1 = pin_robot.get_ee_pose()
    pin_pos, pin_quat, pin_q = ik_helper(pin_robot, pos1, quat1)
    pos1, quat1 = pb_robot.get_ee_pose()
    pb_pos, pb_quat, pb_q = ik_helper(pb_robot, pos1, quat1)
    pos_err = compute_err(pin_pos, pb_pos)
    quat_err = quaternion_distance(pin_quat, pb_quat)
    assert pos_err < POS_ERROR_TOL
    assert quat_err < ORI_ERROR_TOL


def test_pinocchio_optimize_against_pybullet_optimize(
    pin_optimize_robot, pb_optimize_robot, test_pose
):
    np.random.seed(0)
    pin_optimize_robot.set_config(STRETCH_HOME_Q)
    pb_optimize_robot.set_config(STRETCH_HOME_Q)

    # Make the orientation bounds looser because for optimization, we're allowing some slack in orientation
    ori_error_tol = 0.1

    # Set state to home pose
    pos, quat = test_pose

    # Run ik
    print("-------- 1: Inverse kinematics ---------")
    pin_pos, pin_quat, pin_q = ik_helper(
        pin_optimize_robot, pos, quat, ori_error_tol=ori_error_tol
    )
    pb_pos, pb_quat, pb_q = ik_helper(
        pb_optimize_robot, pos, quat, ori_error_tol=ori_error_tol
    )
    print(f"Pinocchio: {pin_pos}, {pin_quat}, {pin_q}")
    print(f"PyBullet: {pb_pos}, {pb_quat}, {pb_q}")
    pos_err = compute_err(pin_pos, pb_pos)
    assert pos_err < POS_ERROR_TOL

    print("-------- 2: FK + IK Consistency  ---------")
    pos1, quat1 = pin_optimize_robot.get_ee_pose()
    pin_pos, pin_quat, pin_q = ik_helper(
        pin_optimize_robot, pos1, quat1, ori_error_tol=ori_error_tol
    )
    pos1, quat1 = pb_optimize_robot.get_ee_pose()
    pb_pos, pb_quat, pb_q = ik_helper(
        pb_optimize_robot, pos1, quat1, ori_error_tol=ori_error_tol
    )
    pos_err = compute_err(pin_pos, pb_pos)
    assert pos_err < POS_ERROR_TOL


def test_pinocchio_ik_optimization(pin_robot, pin_ik_optimizer, test_pose):
    pos_desired = np.array(test_pose[0])
    quat_desired = np.array(test_pose[1])

    # Directly solve with IK
    q, success, pin_debug_info = pin_robot.manip_ik_solver.compute_ik(
        pos_desired, quat_desired
    )
    pos_out1, quat_out1 = pin_robot.manip_ik_solver.compute_fk(q)
    pos_err1 = np.linalg.norm(pos_out1 - pos_desired)

    # Solve with CEM
    q_result, success, pin_optimizer_debug_info = pin_ik_optimizer.compute_ik(
        pos_desired, quat_desired
    )
    best_cost = pin_optimizer_debug_info["best_cost"]
    last_iter = pin_optimizer_debug_info["last_iter"]
    opt_sigma = pin_optimizer_debug_info["opt_sigma"]

    pos_out2, quat_out2 = pin_robot.manip_ik_solver.compute_fk(q_result)
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
        best_cost <= pin_ik_optimizer.opt.cost_tol
        or np.all(opt_sigma) <= pin_ik_optimizer.opt.cost_tol
        or last_iter >= pin_ik_optimizer.opt.max_iterations
    )
    assert success


def test_pybullet_ik_optimization(pb_robot, pb_ik_optimizer, test_pose):
    np.random.seed(0)
    pos_desired = np.array(test_pose[0])
    quat_desired = np.array(test_pose[1])

    # Directly solve with IK
    q, success, pin_debug_info = pb_robot.manip_ik_solver.compute_ik(
        pos_desired, quat_desired
    )
    pos_out1, quat_out1 = pb_robot.manip_ik_solver.compute_fk(q)
    pos_err1 = np.linalg.norm(pos_out1 - pos_desired)

    # Solve with CEM
    q_result, success, pin_optimizer_debug_info = pb_ik_optimizer.compute_ik(
        pos_desired, quat_desired
    )
    pos_out2, quat_out2 = pb_robot.manip_ik_solver.compute_fk(q_result)
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
    assert success


def test_ros_to_pin(pin_robot, test_joints):
    pin_pose = pin_robot._ros_pose_to_pinocchio(test_joints[0])
    assert len(pin_pose) == len(test_joints[1])
    assert pin_pose == pytest.approx(test_joints[1])


if __name__ == "__main__":
    robot_model = HelloStretchKinematics(
        urdf_path=URDF_ABS_PATH,
        visualize=DEBUG,
        ik_type="pybullet",
    )
    opt = PositionIKOptimizer(
        robot_model.manip_ik_solver,
        pos_error_tol=CEM_POS_ERROR_TOL,
        ori_error_range=np.array([0.0, 0.0, CEM_YAW_ERROR_TOL]),  # solve for yaw only
    )
    test_ik_solvers(robot_model, TEST_DATA[0])
    test_pybullet_ik_optimization(robot_model, opt, TEST_DATA[0])
