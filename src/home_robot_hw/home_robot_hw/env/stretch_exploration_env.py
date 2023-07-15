# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.motion.stretch import STRETCH_NAVIGATION_Q, HelloStretchKinematics
from home_robot.utils.config import get_config
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global
from home_robot_hw.constants import relative_resting_position
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import ExplorationVisualizer
from home_robot_hw.remote import StretchClient


class StretchExplorationEnv(StretchEnv):
    """Create an exploration environment"""

    def __init__(self, config, visualize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.forward_step = config.ENVIRONMENT.forward  # in meters
        self.rotate_step = np.radians(config.ENVIRONMENT.turn_angle)
        self.max_steps = config.ENVIRONMENT.max_steps
        self.curr_step = None

        if visualize:
            self.visualizer = ExplorationVisualizer(config)
        else:
            self.visualizer = None

        # Create a robot model, but we never need to visualize
        self.robot = StretchClient(init_node=False)
        self.robot_model = self.robot.robot_model
        self.reset()

    def reset(self):
        """Save start pose and reset everything."""
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        self.curr_step = 0
        if self.visualizer is not None:
            self.visualizer.reset()
        self.robot.move_to_nav_posture()

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Discrete action space. make predictions for where the robot should go, move by a fixed
        amount forward or rotationally."""
        self.curr_step += 1
        if self.visualizer is not None:
            self.visualizer.visualize(**info)
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
        else:
            # Do nothing if "stop"
            # continuous_action = None
            # if not self.in_manipulation_mode():
            #     self.switch_to_manipulation_mode()
            pass

        if continuous_action is not None:
            if not self.robot.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            if not self.dry_run:
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=True
                )

    def get_observation(self) -> Observations:
        """Get rgb/xyz/theta from this"""
        rgb, depth, xyz = self.robot.head.get_images(compute_xyz=True)
        current_pose = xyt2sophus(self.robot.nav.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Get joint state information
        joint_positions, _, _ = self.robot.get_joint_state()

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=gps,
            compass=np.array([theta]),
            task_observations={"image_frame": rgb.copy()[:, :, ::-1]},
            camera_pose=self.robot.head.get_pose(rotated=True),
            joint=self.robot.model.config_to_hab(joint_positions),
            relative_resting_position=relative_resting_position,
        )
        return obs

    @property
    def episode_over(self) -> bool:
        return self.curr_step >= self.max_steps

    def get_episode_metrics(self) -> Dict:
        pass

    def rotate(self, theta):
        """just rotate and keep trying"""
        init_pose = self.robot.nav.get_base_pose()
        xyt = [0, 0, theta]
        goal_pose = xyt_base_to_global(xyt, init_pose)
        rate = rospy.Rate(5)
        err = float("Inf"), float("Inf")
        pos_tol, ori_tol = 0.1, 0.1
        while not rospy.is_shutdown():
            curr_pose = self.robot.nav.get_base_pose()
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

    config_path = "projects/stretch_exploration/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)

    rob = StretchExplorationEnv(config, init_cameras=True)
    rob.robot.switch_to_navigation_mode()

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
