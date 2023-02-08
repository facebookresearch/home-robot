import home_robot
import numpy as np
import rospy

from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, Observations
from home_robot_hw.env.stretch_abstract_env import StretchEnv
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import sophus2obs, obs2xyt


REAL_WORLD_CATEGORIES=["chair", "mug"]


class StretchObjectNavEnv(StretchEnv):
    """ Create a detic-based object nav environment""" 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: pass this in or load from cfg
        self.goal_options = REAL_WORLD_CATEGORIES

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(self.goal_options),
            sem_gpu_id=0,
        )
        self.reset()

    def reset(self):
        self.sample_goal()
        self._episode_start_pose = self.get_base_pose()

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        pass

    def set_goal(self, goal):
        """ set a goal as a string"""
        if goal in self.goal_options:
            self.current_goal_id = self.goal_options.index(goal)
            self.current_goal_name = goal
            return True
        else:
            return False

    def sample_goal(self):
        """ set a random goal """
        idx = np.random.randint(len(self.goal_options))
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        rgb, depth = self.get_images(compute_xyz=False, rotate_images=True)
        current_pose = self.get_base_pose()

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        print(relative_pose)
        print("xyz =", relative_pose.translation())
        print("rpy", euler_angles)

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb,
            depth=depth,
            base_pose=sophus2obs(relative_pose)
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
            }
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        return obs

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass



if __name__ == '__main__':
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

    xyt = obs2xyt(obs.base_pose)
    xyt[0] += 0.1
    rob.navigate_to(xyt)
    rospy.sleep(5.)
    obs = rob.get_observation()
    observations.append(obs)

    xyt[0] = 0
    rob.navigate_to(xyt)
    rospy.sleep(5.)
    obs = rob.get_observation()
    observations.append(obs)

    for obs in observations:
        rgb, depth = obs.rgb, obs.depth
        xyt = obs2xyt(obs.base_pose)

        # Add a visualiztion for debugging
        import matplotlib.pyplot as plt
        depth[depth > 5] = 0
        plt.subplot(121); plt.imshow(rgb)
        plt.subplot(122); plt.imshow(depth)

        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        print("XY =", xyt[:2])
        print("Yaw=", xyt[-1])
        plt.show()
