from typing import Dict

import numpy as np
import trimesh
from home_robot.motion.stretch import (STRETCH_BASE_FRAME, STRETCH_GRASP_FRAME,
                                       HelloStretch)
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchManipulationEnv(StretchEnv):
    """Simple environment to enable rolling out manipulation policies via SLAP"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot = HelloStretch(ik_type="pinocchio")

    def reset(self) -> None:
        """Reset is called at the beginning of each episode where the
        robot retracts gripper and sets itself to a neutral position"""
        # TODO: implement this
        pass

    def apply_action(self, manip_action) -> None:
        """
        manip_action: Manipulation action in cartesian space
                      (pos, quat)
        """
        if manip_action is None:
            manip_action = self.get_pose(STRETCH_GRASP_FRAME, STRETCH_BASE_FRAME)
        q0, _ = self.update()
        q = self.robot.manip_ik(manip_action, q0=q0)
        self.goto(q, wait=True, move_base=True)
        print("Moved to predicted action")

    # over-riding the following methods from the parent class
    def episode_over(self) -> None:
        pass

    def get_episode_metrics(self) -> None:
        pass

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Collects sensor data and passes as a dictionary to SLAP
        1. rgb image
        2. depth image
        3. base pose
        4. joint positions
        5. joint velocities
        6. camera pose
        7. camera info
        8. end-effector pose
        """
        # record rgb and depth
        camera_pose = self.get_camera_pose_matrix()
        rgb, depth, xyz = self.get_images(compute_xyz=True, rotate_images=True)
        q, dq = self.update()
        # apply depth filter
        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        cam_xyz = xyz.reshape(-1, 3)
        xyz = trimesh.transform_points(cam_xyz, camera_pose)
        valid_depth = np.bitwise_and(depth > 0.1, depth < 4.0)
        rgb = rgb[valid_depth, :]
        xyz = xyz[valid_depth, :]
        show_point_cloud(xyz, rgb / 255.0, orig=np.zeros(3))
        # TODO get the following from TF lookup
        # look up TF of link_straight_gripper wrt base_link
        # add grasp-offset to this pose
        ee_pose_0 = self.robot.fk(q)
        breakpoint()
        ee_pose_1 = self.get_pose("link_grasp_center", base_frame=STRETCH_BASE_FRAME)
        # QUESTION: what is the difference between above two?
        # output of above is a tuple of two ndarrays
        # ee-pose should be 1 ndarray of 7 values
        ee_pose = np.concatenate((ee_pose_0[0], ee_pose_0[1]), axis=0)
        # elements in following are of type: Tuple(Tuple(x,y,theta), rospy.Time)
        # change to ndarray with 4 floats
        base_pose = self.get_base_pose()
        breakpoint()
        observations = {
            "rgb": rgb,
            "depth": depth,
            "q": q,
            "dq": dq,
            "ee_pose": ee_pose,
            "camera_pose": camera_pose,
            "camera_info": self.get_camera_info(),
            "base_pose": base_pose,
        }
        return observations
