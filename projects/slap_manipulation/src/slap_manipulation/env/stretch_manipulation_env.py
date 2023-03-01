from typing import Dict

import numpy as np
from home_robot.motion.stretch import HelloStretch, STRETCH_BASE_FRAME, STRETCH_GRASP_FRAME
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchManipulationEnv(StretchEnv):
    """Simple environment to enable rolling out manipulation policies via SLAP"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot = HelloStretch(ik_type="pinocchio")

    def reset(self) -> None:
        """Reset is called at the beginning of each episode where the
        robot drives back for 0.5m and then resets to a neutral position"""
        # TODO: implement this
        pass

    def apply_action(self, manip_action) -> None:
        """
        manip_action: Manipulation action in cartesian space
                      (pos, quat)
        """
        q0, _ = self.update()
        q = self.robot.manip_ik(manip_action, q0=q0)
        self.goto(q, wait=True, move_base=True)
        print("Moved to predicted action")

    # over-riding the following methods from the parent class
    def episode_over(self) -> None:
        pass

    def get_episode_metrics(self) -> None:
        pass

    def get_observation(self) -> Dict[str]:
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
        rgb, depth = self.get_images(compute_xyz=False)
        q, dq = self.update()
        # TODO get the following from TF lookup
        # look up TF of link_straight_gripper wrt base_link
        # add grasp-offset to this pose
        ee_pose_0 = self.robot.fk(q)
        ee_pose_1 = self.get_pose(STRETCH_GRASP_FRAME, STRETCH_BASE_FRAME)
        # QUESTION: what is the difference between above two?
        # output of above is a tuple of two ndarrays
        # ee-pose should be 1 ndarray of 7 values
        ee_pose = np.concatenate((ee_pose[0], ee_pose[1]), axis=0)
        # elements in following are of type: Tuple(Tuple(x,y,theta), rospy.Time)
        # change to ndarray with 4 floats
        base_pose = self.get_base_pose()
        base_pose = np.array(
            [base_pose[0][0], base_pose[0][1], base_pose[0][2], base_pose[1].to_sec()]
        )
        camera_pose = self.get_camera_pose_matrix()
        observations = {
            "rgb": rgb,
            "depth": depth,
            "q": q,
            "dq": dq,
            "ee_pose": ee_pose,
            "camera_pose": camera_pose,
            "camera_info": self.get_camera_info(),
        }
        return observations
