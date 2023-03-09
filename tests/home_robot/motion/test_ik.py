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

POS_ERROR_TOL = 1e-4  # 0.1 mm
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


# Helper functions


def compute_err(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def quaternion_distance(quat1, quat2):
    """
    Returns: (1 - cos(theta_diff)) / 2
    """
    return 1 - ((quat1 * quat2).sum() ** 2)


def ik_helper(robot, pos, quat, indicator_block=None, debug=DEBUG):
    """ik test helper function."""
    print("GOAL:", pos, quat)
    if indicator_block is not None:
        indicator_block.set_pose(pos, quat)
    res = robot.manip_ik((pos, quat), STRETCH_HOME_Q, relative=True)
    robot.set_config(res)
    pos2, quat2 = robot.get_ee_pose()
    print("RESULT:", pos2, quat2)
    print("x motion:", res[0])
    err = compute_err(pos2, pos)
    print("error was:", err)
    assert err < POS_ERROR_TOL
    assert quaternion_distance(quat, quat2) < ORI_ERROR_TOL
    if debug:
        input("press enter to continue")
    return pos2, quat2, res


# Tests


@pytest.fixture(params=TEST_DATA)
def test_pose(request):
    pos_raw, quat_raw = request.param
    pos, quat = to_pos_quat(to_matrix(pos_raw, quat_raw) @ STRETCH_GRASP_OFFSET)
    return pos, quat


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


@pytest.fixture(params=["pybullet", "pinocchio"])
def robot(request, pb_robot, pin_robot):
    if request.param == "pybullet":
        return pb_robot
    elif request.param == "pinocchio":
        return pin_robot


@pytest.fixture
def pin_ik_optimizer(pin_robot):
    return PositionIKOptimizer(
        pin_robot.manip_ik_solver,
        pos_error_tol=CEM_POS_ERROR_TOL,
        ori_error_range=np.array([0.0, 0.0, CEM_YAW_ERROR_TOL]),  # solve for yaw only
    )


def test_ik_solvers(robot, test_pose):
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
    ik_helper(robot, pos, quat, block)

    print("-------- 2: FK + IK Consistency  ---------")
    pos1, quat1 = robot.get_ee_pose()
    ik_helper(robot, pos1, quat1, block)


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


def test_pinocchio_ik_optimization(pin_robot, pin_ik_optimizer, test_pose):
    pos_desired = np.array(test_pose[0])
    quat_desired = np.array(test_pose[1])

    # Directly solve with IK
    q, _ = pin_robot.manip_ik_solver.compute_ik(pos_desired, quat_desired)
    pos_out1, quat_out1 = pin_robot.manip_ik_solver.compute_fk(q)
    pos_err1 = np.linalg.norm(pos_out1 - pos_desired)

    # Solve with CEM
    q_result, best_cost, last_iter, opt_sigma = pin_ik_optimizer.compute_ik_opt(
        (pos_desired, quat_desired)
    )
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
