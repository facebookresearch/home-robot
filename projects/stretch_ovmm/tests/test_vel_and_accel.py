#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This script is used to run an experiment on the Stretch robot with different velocities and accelerations profiles.
"""

import click
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import TransformStamped

from home_robot.agent.hierarchical.pick_and_place_agent import PickAndPlaceAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.motion.stretch import (
    STRETCH_HOME_Q,
    STRETCH_NAVIGATION_Q,
    STRETCH_PREGRASP_Q,
)
from home_robot.utils.config import get_config
from home_robot.utils.geometry import posquat2sophus, sophus2posquat, xyt2sophus
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import (
    StretchPickandPlaceEnv,
    load_config,
)
from home_robot_hw.ros.utils import matrix_to_pose_msg, ros_pose_to_transform

motion_choices = ["default", "slow", "fast", "very_slow"]


def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_data


def get_velocities(motion_profiles_dict, joint_names, speed: str = "default"):
    assert (
        speed in motion_choices
    ), "Speed must be 'default', 'slow', 'fast', or 'very_slow'"
    joint_velocities = []
    for joint in joint_names:
        joint_velocities.append(motion_profiles_dict[joint][speed]["vel"])
    return joint_velocities


def get_accelerations(motion_profiles_dict, joint_names, speed: str = "default"):
    assert (
        speed in motion_choices
    ), "Speed must be 'default', 'slow', 'fast', or 'very_slow'"
    joint_accelerations = []
    for joint in joint_names:
        joint_accelerations.append(motion_profiles_dict[joint][speed]["accel"])
    return joint_accelerations


def run(
    env,
    pose: np.ndarray,
    motion_profiles_dict: dict,
    joints_names: list,
    speed: str = "default",
):
    joint_poses = np.array(pose)
    joint_velocities = get_velocities(motion_profiles_dict, joints_names, speed)
    assert len(joint_velocities) == len(
        joint_poses
    ), "Number of joint velocities does not match number of joints."
    joint_accelerations = get_accelerations(motion_profiles_dict, joints_names, speed)
    assert len(joint_accelerations) == len(
        joint_poses
    ), "Number of joint accelerations does not match number of joints."
    env.robot.manip.goto(
        joint_poses, dq=joint_velocities, ddq=joint_accelerations, wait=True
    )


@click.command()
@click.option("--visualize-maps", default=False, is_flag=True)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--test-id", default=0, type=int)
def run_experiment(visualize_maps=False, test_id=0, reset_nav=False, **kwargs):
    config = load_config(visualize=visualize_maps, **kwargs)
    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(config=config)
    env.reset("table", "cup", "chair")
    robot = env.get_robot()

    if reset_nav:
        # Send it back to origin position to make testing a bit easier
        robot.nav.navigate_to([0, 0, 0])

    # Put it into initial posture
    env.robot.move_to_manip_posture()
    # # Go to stow position with default motion profile
    # env.robot.manip.goto(STRETCH_HOME_Q, dq=None, ddq=None, wait=True)

    # Get motion profiles
    root = "src/home_robot/config/control/"
    motion_profiles_dict = read_yaml(root + "motion_profiles.yaml")

    # Define test pose and motion profile
    test_motion = "very_slow"
    pose = STRETCH_HOME_Q
    joints_for_mp = [
        "base",
        "base",
        "base",
        "lift",
        "arm",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
        "stretch_gripper",
        "head_pan",
        "head_tilt",
    ]

    # run experiment
    run(env, pose, motion_profiles_dict, joints_for_mp, test_motion)


if __name__ == "__main__":
    run_experiment()
