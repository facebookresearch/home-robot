from typing import Any, Dict, Optional

import os
import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot_hw.env.visualizer import ExplorationVisualizer


class StretchExplorationEnv(StretchEnv):
    """Create an exploration environment for occupancy mapping"""

    def __init__(
        self, config=None, forward_step=0.25, rotate_step=30.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        if config is not None:
            self.visualizer = ExplorationVisualizer(config)
        else:
            self.visualizer = None
        self.reset()

    def reset(self):
        self._episode_start_pose = xyt2sophus(self.get_base_pose())
        if self.visualizer is not None:
            self.visualizer.reset()

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
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
                self.switch_to_navigation_mode()
                rospy.sleep(self.msg_delay_t)
            self.navigate_to(continuous_action, relative=True, blocking=True)
        rospy.sleep(0.5)

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
        )
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
    rob = StretchExplorationEnv(init_cameras=True)
    rob.switch_to_navigation_mode()

    save_dir = '/home/santhosh/Research/transparent_images_dataset'
    os.makedirs(save_dir, exist_ok=True)

    image_count = 0
    # Debug the observation space
    import matplotlib.pyplot as plt

    rgb, depth = None, None
    while not rospy.is_shutdown():

        while not rospy.is_shutdown():
            cmd = None
            try:
                cmd = input("Enter a number 1-4:")
                cmd = int(cmd)
                assert cmd in [1, 2, 3, 4]
                if cmd == 4:
                    cmd = None
                    if rgb is not None:
                        rgb_save_path = os.path.join(save_dir, f"rgb_{image_count}.npy")
                        depth_save_path = os.path.join(save_dir, f"depth_{image_count}.npy")
                        np.save(rgb_save_path, rgb)
                        np.save(depth_save_path, depth)
                        image_count += 1
                else:
                    cmd = DiscreteNavigationAction(cmd)
            except ValueError:
                cmd = None
            if cmd is not None:
                break
        if cmd is not None:
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
