# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio
import numpy as np
import torch

from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.core.robot import ControlMode, GraspClient, RobotClient
from home_robot.motion.robot import RobotModel
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.image import Camera
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)


class OvmmSimClient(RobotClient):
    """Defines the ovmm simulation robot as a RobotClient child
    class so the sim can be used with the cortex demo code"""

    _success_tolerance = 1e-4

    def __init__(
        self,
        sim_env: HabitatOpenVocabManipEnv,
        is_stretch_robot: bool,
    ):
        super().__init__()

        self.env = sim_env
        # self.obs = self.env.reset() if running only one episode

        self._last_motion_failed = False

        self.done = False
        self.hab_info = None

        self.video_frames = []
        self.fpv_video_frames = []

        if is_stretch_robot:
            self._robot_model = HelloStretchKinematics(
                urdf_path="",
                ik_type="pinocchio",
                visualize=False,
                grasp_frame=None,
                ee_link_name=None,
                manip_mode_controlled_joints=None,
            )
        self.num_action_applied = 0
        self.force_quit = False

    def navigate_to(
        self,
        xyt: ContinuousNavigationAction,
        relative: bool = False,
        blocking: bool = False,
        verbose: bool = False,
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        if not relative:
            xyt = xyt_global_to_base(xyt, self.get_base_pose())

        if type(xyt) != np.ndarray:
            xyt = np.array(xyt)

        xyt = ContinuousNavigationAction(xyt)
        if verbose:
            print("NAVIGATE TO", xyt.xyt, relative, blocking)

        self.apply_action(xyt, verbose=verbose)

    def reset(self):
        """Reset everything in the robot's internal state"""
        self.obs = self.env.reset()
        self.video_frames = [self.obs.third_person_image]
        self.fpv_video_frames = [self.obs.rgb]
        self.num_action_applied = 0
        self.done = False
        self.force_quit = False

    def switch_to_navigation_mode(self) -> bool:
        """Apply sim navigation mode action and set internal state"""
        self.apply_action(DiscreteNavigationAction.NAVIGATION_MODE)
        self._base_control_mode = ControlMode.NAVIGATION

        return True

    def switch_to_manipulation_mode(self) -> bool:
        """Apply sim manipulation mode action and set internal state"""
        self.apply_action(DiscreteNavigationAction.MANIPULATION_MODE)
        self._base_control_mode = ControlMode.MANIPULATION

        return True

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model

    def get_observation(self):
        """Return obs from last apply action"""
        return self.obs

    def get_task_obs(self) -> Tuple[str, str]:
        """Return object_to_find and location_to_place"""
        return (
            self.obs.task_observations["object_name"],
            self.obs.task_observations["start_recep_name"],
            self.obs.task_observations["place_recep_name"],
        )

    def move_to_nav_posture(self):
        """No applicable action in sim"""
        self.apply_action(DiscreteNavigationAction.EMPTY_ACTION)

    def move_to_manip_posture(self):
        """No applicable action in sim"""
        self.apply_action(DiscreteNavigationAction.EMPTY_ACTION)

    def get_base_pose(self):
        """xyt position of robot"""
        return np.array([self.obs.gps[0], self.obs.gps[1], self.obs.compass[0]])

    def episode_over(self):
        return (
            self.num_action_applied
            >= self.env.config["habitat"]["environment"]["max_episode_steps"]
        )

    def apply_action(self, action, verbose: bool = False):
        verbose = True
        """Actually send the action to the simulator."""
        if self.episode_over():
            print("habitat env is closed, so the robot can't take any actions")
            self.force_quit = True
            return
        xyt0 = self.get_base_pose()
        if verbose:
            print("STARTED AT:", xyt0)
            print("ACTION:", action)

        # constraints
        if isinstance(action, ContinuousNavigationAction):
            for axis, delta in enumerate(action.xyt[:2]):
                if abs(delta) < 0.1 and delta != 0:
                    print(
                        "the robot is trying to make tiny movement along an axis, set it to 0"
                    )
                    action.xyt[axis] = 0.0
            if abs(action.xyt[2]) <= math.radians(5) and abs(action.xyt[2]) > 1e-8:
                print("the robot is trying to rotate by a tiny angle, set it to 0")
                action.xyt[2] = 0.0
            # return
            if verbose:
                print("NEW ACTION:", action)

        self.obs, self.done, self.hab_info = self.env.apply_action(action)
        self.num_action_applied += 1
        if verbose:
            print("MOVED TO:", self.get_base_pose())
        xyt1 = self.get_base_pose()

        # if these are the same within some tolerance, the motion failed
        if isinstance(action, ContinuousNavigationAction):
            large_action = np.linalg.norm(action.xyt) > self._success_tolerance
            self._last_motion_failed = (
                large_action and np.linalg.norm(xyt0 - xyt1) < self._success_tolerance
            )
        else:
            self._last_motion_failed = True
        self.video_frames.append(self.obs.third_person_image)
        self.fpv_video_frames.append(self.obs.rgb)

        # self.save_frame()

    def last_motion_failed(self):
        return self._last_motion_failed

    def save_frame(self):
        """Save frame for debug the sim client at each step"""
        imageio.imwrite(
            os.path.join(self.debug_path, str(time.time()) + ".png"),
            self.obs.third_person_image,
        )

    def make_video(self, path=os.getcwd(), name="debug.mp4"):
        """Save a video for this sim client"""
        imageio.mimsave(os.path.join(path, name), self.video_frames, fps=30)

    def make_fpv_video(self):
        """Save a fpv video for this sim client"""
        imageio.mimsave("debug_fpv.mp4", self.fpv_video_frames, fps=30)

    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
    ):
        """Execute a multi-step trajectory by making looping calls to navigate_to"""
        for i, pt in enumerate(trajectory):
            assert (
                len(pt) == 3 or len(pt) == 2
            ), "base trajectory needs to be 2-3 dimensions: x, y, and (optionally) theta"
            self.navigate_to(pt, relative, verbose=verbose)
            if self.last_motion_failed():
                return False
        return True

    # def get_metrics(self):
    #     self.hab_info


class SimGraspPlanner(GraspClient):
    """Interface to simulation grasping"""

    def __init__(
        self,
        robot_client: RobotClient,
    ):
        self.robot_client = robot_client

    def set_robot_client(self, robot_client: RobotClient):
        """Update the robot client this grasping client uses"""
        self.robot_client = robot_client

    def try_grasping(self, object_goal: Optional[str] = None) -> bool:
        """Grasp the object by snapping object in sim"""
        if object_goal:
            pass
        self.robot_client.apply_action(DiscreteNavigationAction.SNAP_OBJECT)
        return True

    def try_placing(self, object_goal: Optional[str] = None) -> bool:
        """Place the object by de-snapping object in sim"""
        self.robot_client.apply_action(DiscreteNavigationAction.DESNAP_OBJECT)
        return True
