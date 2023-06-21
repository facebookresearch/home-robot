# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import cv2
import numpy as np
import rospy
from omegaconf import DictConfig

import home_robot
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchImageNavEnv(StretchEnv):
    """Create a detic-based object nav environment"""

    def __init__(
        self, config: Optional[DictConfig] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if config:
            self.forward_step = config.habitat.simulator.forward_step_size  # in meters
            self.rotate_step = np.radians(config.habitat.simulator.turn_angle)
            self.image_goal = self._load_image_goal(config.stretch_goal_image_path)
        else:
            self.forward_step = 0.25
            self.rotate_step = np.radians(30)
            self.image_goal = None
        self.reset()

    def _load_image_goal(self, goal_img_path: str) -> np.ndarray:
        """Load the pre-computed image goal from disk."""
        goal_image = cv2.imread(goal_img_path)
        # opencv loads as BGR, but we use RGB.
        goal_image = goal_image[:, :, ::-1]
        assert goal_image.shape[0] == 512
        assert goal_image.shape[1] == 512
        return goal_image

    def reset(self) -> None:
        self._episode_start_pose = xyt2sophus(self.get_base_pose())
        self.goto(STRETCH_HOME_Q)

    def apply_action(self, action: DiscreteNavigationAction) -> None:
        """Convert a DiscreteNavigationAction to a continuous action and perform it"""
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            print("FORWARD")
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            print("TURN RIGHT")
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            print("TURN LEFT")
            continuous_action[2] = self.rotate_step
        elif action == DiscreteNavigationAction.STOP:
            print("Done!")
        else:
            raise RuntimeError("Action type not supported: " + str(action))

        if not self.in_navigation_mode():
            self.switch_to_navigation_mode()
        self.navigate_to(continuous_action, relative=True, blocking=True)

    def get_observation(self) -> Observations:
        """Get rgbd/xyz/theta from this"""
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
        current_pose = xyt2sophus(self.get_base_pose())

        # Gets current camera pose from SLAM system as a 4x4 matrix in SE(3)
        # camera_pose = self.get_camera_pose_matrix()

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        # pos, vel, frc = self.get_joint_state()

        # Create the observation
        return home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=relative_pose.translation()[:2],
            compass=np.array([theta]),
            task_observations={"instance_imagegoal": self.image_goal},
            # camera_extrinsic=camera_pose,
        )

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass

    def rotate(self, theta: float) -> None:
        """just rotate and keep trying"""
        init_pose = self.get_base_pose()
        xyt = [0, 0, theta]
        goal_pose = xyt_base_to_global(xyt, init_pose)
        rate = rospy.Rate(5)
        err = float("Inf"), float("Inf")
        pos_tol, ori_tol = 0.1, 0.1
        while not rospy.is_shutdown():
            curr_pose = self.get_base_pose()
            print("init =", init_pose)
            print("curr =", curr_pose)
            print("goal =", goal_pose)

            print("error =", err)
            if err[0] < pos_tol and err[1] < ori_tol:
                break
            rate.sleep()


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchImageNavEnv(init_cameras=True)
    rob.switch_to_navigation_mode()

    # Debug the observation space
    import matplotlib.pyplot as plt

    while not rospy.is_shutdown():

        while not rospy.is_shutdown():
            cmd = None
            try:
                cmd = input("Enter a number 0-3:")
                cmd = DiscreteNavigationAction(int(cmd))
            except ValueError:
                cmd = None
            if cmd is not None:
                break
        rob.apply_action(cmd)

        obs = rob.get_observation()
        rgb, depth = obs.rgb, obs.depth
        # xyt = obs2xyt(obs.base_pose)

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)
        # plt.subplot(133); plt.imshow(obs.semantic

        print()
        print("----------------")
        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        # print("XY =", xyt[:2])
        # print("Yaw=", xyt[-1])
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()


if False:
    observations = []
    obs = rob.get_observation()
    observations.append(obs)

    xyt = np.zeros(3)
    xyt[2] = obs.compass
    xyt[:2] = obs.gps
    # xyt = obs2xyt(obs.base_pose)
    xyt[0] += 0.1
    # rob.navigate_to(xyt)
    rob.rotate(0.2)
    rospy.sleep(10.0)
    obs = rob.get_observation()
    observations.append(obs)

    xyt[0] = 0
    # rob.navigate_to(xyt)
    rob.rotate(-0.2)
    rospy.sleep(10.0)
    obs = rob.get_observation()
    observations.append(obs)

    for obs in observations:
        rgb, depth = obs.rgb, obs.depth
        # xyt = obs2xyt(obs.base_pose)

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)
        # plt.subplot(133); plt.imshow(obs.semantic

        print()
        print("----------------")
        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        # print("XY =", xyt[:2])
        # print("Yaw=", xyt[-1])
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()
