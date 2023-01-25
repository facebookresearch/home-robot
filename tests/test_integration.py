import pytest
import signal
import time
from subprocess import Popen

import numpy as np

from home_robot.client.local_hello_robot import LocalHelloRobot


MAX_RETRIES = 10


@pytest.fixture()
def home_robot_stack():
    p_roscore = Popen(["roscore"])
    p_sim = Popen(["python", "-m", "home_robot_sim.nodes.fake_stretch_robot"])
    p_se = Popen(["python", "-m", "home_robot.nodes.state_estimator"])
    p_gc = Popen(["python", "-m", "home_robot.nodes.goto_controller"])

    return [p_roscore, p_sim, p_se, p_gc]

@pytest.fixture()
def robot():
    time.sleep(1) # HACK: wait for stack to launch

    retries = 0
    while retries < MAX_RETRIES:
        try:
            robot = LocalHelloRobot()
            if robot.get_base_state() is None:
                continue
            else:
                break
        except Exception:
            time.sleep(0.5)

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

    for p in home_robot_stack[::-1]:
        p.send_signal(signal.SIGINT)
