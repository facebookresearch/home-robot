import pytest
import time

import numpy as np
import mrp

from home_robot.client import LocalHelloRobot


@pytest.fixture()
def home_robot_stack():
    mrp.import_msetup("../src/home_robot")
    mrp.cmd.up("sim_stack")


@pytest.fixture()
def robot():
    robot = LocalHelloRobot()

    # HACK: wait for controller to launch
    while robot.get_base_state() is None:
        time.sleep(0.2)

    return robot


def test_goto(home_robot_stack, robot):
    xyt_goal = [0.2, 0.1, 0.1]

    # Activate goto controller & set goal
    robot.toggle_controller()
    robot.set_goal(xyt_goal)

    # Wait for robot to reach goal
    time.sleep(3)

    # Check that robot is at goal
    xyt_new = robot.get_base_state()

    assert np.allclose(xyt_new[:2], xyt_goal[:2], atol=0.01)  # 1cm
    assert np.allclose(xyt_new[2], xyt_goal[2], atol=0.052)  # 3 degrees

    # Down processes
    mrp.cmd.down()
