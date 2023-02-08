import home_robot
import numpy as np
import rospy

from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations, DiscreteNavigationAction
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import sophus2obs, obs2xyt
from home_robot_hw.env.visualizer import Visualizer


# REAL_WORLD_CATEGORIES = ["chair", "mug"]
REAL_WORLD_CATEGORIES = ["backpack"]


class StretchObjectNavEnv(StretchEnv):
    """Create a detic-based object nav environment"""

    def __init__(self, config=None, forward_step=0.25, rotate_step=30., *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(self.goal_options) + ",other",
            sem_gpu_id=0,
        )
        if config is not None:
            self.visualizer = Visualizer(config)
        else:
            self.visualizer = None
        self.reset()

    def reset(self):
        self.sample_goal()
        self._episode_start_pose = self.get_base_pose()
        if self.visualizer is not None:
            self.visualizer.reset()

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        if self.visualizer is not None:
            self.visualizer.visualize(**info)
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
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
            self.navigate_to(continuous_action, relative=True)
        print("-------")
        print(action)
        print(continuous_action)
        rospy.sleep(5.)

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
        idx = np.random.randint(len(self.goal_options))
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
        current_pose = self.get_base_pose()

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        pos, vel, frc = self.get_joint_state()

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            gps=relative_pose.translation()[:2],
            compass=np.array([theta]),
            # base_pose=sophus2obs(relative_pose),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
            },
            joint_positions=pos,
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass

    def rotate(self, theta):
        """ just rotate and keep trying"""
        # init_pose = self.get_base_pose()
        init_pose = sophus2xyt(self.get_base_pose())
        xyt = [0, 0, theta]
        goal_pose = xyt_base_to_global(xyt, init_pose)
        rate = rospy.Rate(5)
        err = float('Inf'), float('Inf')
        pos_tol, ori_tol = 0.1, 0.1
        while not rospy.is_shutdown():
            # curr_pose = self.get_base_pose()
            curr_pose = sophus2xyt(self.get_base_pose())
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

    observations = []
    obs = rob.get_observation()
    observations.append(obs)

    xyt = np.zeros(3)
    xyt[2] = obs.compass
    xyt[:2] = obs.gps
    # xyt = obs2xyt(obs.base_pose)
    xyt[0] += 0.1
    #rob.navigate_to(xyt)
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

    # Debug the observation space
    import matplotlib.pyplot as plt

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
