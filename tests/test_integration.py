import pytest
import os
import time
import subprocess

import numpy as np

import home_robot


MAX_RETRIES = 30


@pytest.fixture()
def home_robot_stack():
    subprocess.run(["roscore"])
    subprocess.run(["python", "-m", "home_robot_sim.nodes.fake_stretch_robot"])
    subprocess.run(["python", "-m", "home_robot.nodes.state_estimator"])
    subprocess.run(["python", "-m", "home_robot.nodes.goto_controller"])


@pytest.fixture()
def robot():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            robot = home_robot.client.LocalHelloRobot()
        except Exception:
            time.sleep(1)
            continue

    # HACK: wait for controller to launch
    while robot.get_base_state() is None:
        time.sleep(0.2)

    return robot


def test_goto(home_robot_stack, robot):
    xyt_goal = [0.1, 0.3, 0.1]

    # Activate goto controller & set goal
    robot.switch_to_navigation_mode()
    robot.navigate_to(xyt_goal)

    # Wait for robot to reach goal
    time.sleep(5)

    # Check that robot is at goal
    xyt_new = robot.get_base_state()["pose_se2"]

    assert np.allclose(xyt_new[:2], xyt_goal[:2], atol=0.05)  # 5cm

    os.exit(0)
