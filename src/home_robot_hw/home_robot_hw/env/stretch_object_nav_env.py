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
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import StretchClient

# REAL_WORLD_CATEGORIES = ["other", "chair", "mug", "other",]
# REAL_WORLD_CATEGORIES = ["other", "backpack", "other",]
REAL_WORLD_CATEGORIES = [
    "other",
    "cup",
    "other",
]


class StretchObjectNavEnv(StretchEnv):
    """Create a detic-based object nav environment"""

    def __init__(
        self, config=None, forward_step=0.25, rotate_step=30.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(self.goal_options),
            sem_gpu_id=0,
        )
        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None

        # Create a robot model, but we never need to visualize
        self.robot = StretchClient(init_node=False)
        self.robot_model = self.robot.robot_model
        self.reset()

    def reset(self):
        self.sample_goal()
        self._episode_start_pose = xyt2sophus(self.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

        # Switch control mode on the robot to nav
        self.robot.switch_to_navigation_mode()
        # put the robot in the correct mode with head facing forward
        home_q = STRETCH_NAVIGATION_Q
        # TODO: get this right
        # tilted
        home_q = self.robot_model.update_look_front(home_q.copy())
        # Flat
        # home_q = self.robot_model.update_look_ahead(home_q.copy())
        self.goto(home_q, move_base=False, wait=True)

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Discrete action space. make predictions for where the robot should go, move by a fixed
        amount forward or rotationally."""
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
            if not self.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            if not self.dry_run:
                self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=True
                )
        rospy.sleep(0.5)

    def set_goal(self, goal):
        """set a goal as a string"""
        if goal in self.goal_options:
            self.current_goal_id = self.goal_options.index(goal)
            self.current_goal_name = goal
            return True
        else:
            return False

    def sample_goal(self):
        """set a random goal"""
        # idx = np.random.randint(len(self.goal_options) - 2) + 1
        idx = 1
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
        current_pose = xyt2sophus(self.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        # pos, vel, frc = self.get_joint_state()

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=gps,
            compass=np.array([theta]),
            # base_pose=sophus2obs(relative_pose),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
                "object_goal": self.current_goal_id,
                "recep_goal": self.current_goal_id,
            },
            camera_pose=self.get_camera_pose_matrix(rotated=True),
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        obs.semantic[obs.semantic == 0] = len(self.goal_options) - 1
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass

    def rotate(self, theta):
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
    rob = StretchObjectNavEnv(init_cameras=True)
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
