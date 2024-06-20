# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from home_robot.motion.rrt import RRT
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.utils.simple_env import SimpleEnv


def _run_simple_env(planner, env, start, goal, visualize: bool = False):
    """Helper function to run planner and start/goal"""
    print("--------------")
    print("Planner =", planner)
    print("Start =", start)
    print("Goal =", goal)
    random.seed(0)
    np.random.seed(0)
    res = planner.plan(start, goal)
    print("Success:", res.success)
    if res.success:
        print("Plan =", [n.state for n in res.trajectory])
    assert res.success, f"Planning failed with {planner}"
    if visualize:
        if res.success:
            env.show([n.state for n in res.trajectory])
        else:
            env.show([start, goal])
    return res


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_rrt_simple_env(start, goal, obs, visualize: bool = False):
    """Test just pure RRT stuff"""
    env = SimpleEnv(obs)
    planner = RRT(env.get_space(), env.validate)
    return _run_simple_env(planner, env, start, goal, visualize)


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_shortcut_rrt_simple_env(start, goal, obs, visualize: bool = False):
    """Test just pure RRT stuff"""
    env = SimpleEnv(obs)
    planner0 = RRT(env.get_space(), env.validate)
    planner1 = Shortcut(planner0)
    res0 = _run_simple_env(planner0, env, start, goal, False)
    res1 = _run_simple_env(planner1, env, start, goal, visualize)
    assert len(res0.trajectory) >= len(
        res1.trajectory
    ), "Shortcut should not make plans longer"


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize: bool = False):
    """Test the connect code"""
    env = SimpleEnv(obs)
    planner0 = RRTConnect(env.get_space(), env.validate)
    planner1 = Shortcut(planner0)
    res0 = _run_simple_env(planner0, env, start, goal, False)
    res1 = _run_simple_env(planner1, env, start, goal, visualize)
    assert len(res0.trajectory) >= len(
        res1.trajectory
    ), "Shortcut should not make plans longer"


@pytest.mark.parametrize(
    "start, goal, obs",
    [
        (np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([0.0, 9.0])),
        (np.array([1.0, 4.0]), np.array([9.0, 9.0]), np.array([1.0, 5.0])),
    ],
)
def test_rrt_connect_simple_env(start, goal, obs, visualize: bool = False):
    """Test the connect code"""
    env = SimpleEnv(obs)
    planner = RRTConnect(env.get_space(), env.validate)
    return _run_simple_env(planner, env, start, goal, visualize)


if __name__ == "__main__":
    # Run a simple test here
    start = np.array([1, 1])
    goal = np.array([9, 9])
    obs = np.array([0, 9])
    # TODO: enable for debugging
    test_rrt_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_simple_env(start, goal, obs, visualize=True)
    test_rrt_connect_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize=True)

    start = np.array([1, 4])
    goal = np.array([9, 9])
    obs = np.array([1, 5])
    # TODO: enable if you want to debug this
    # test_rrt_simple_env(start, goal, obs, visualize=True)
    # test_shortcut_rrt_simple_env(start, goal, obs, visualize=True)
    test_rrt_connect_simple_env(start, goal, obs, visualize=True)
    test_shortcut_rrt_connect_simple_env(start, goal, obs, visualize=True)
