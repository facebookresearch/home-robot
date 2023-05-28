from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy

from home_robot.motion.stretch import HelloStretchIdx, HelloStretchKinematics
from home_robot_hw.remote import StretchClient


class CombinedSLAPPlanner(object):
    """Simple skill motion planner to connect three-six waypoints into a continuous motion"""

    def __init__(self, robot: StretchClient, skill_standoffs: Dict):
        """
        Solve IK
        """
        if not isinstance(robot, StretchClient):
            raise RuntimeError(
                "The SimpleSkillMotionPlanner was designed only for Stretch."
            )
        self.robot = robot
        self.skill_specific_standoffs = skill_standoffs

    def plan_for_skill(self, skill_actions: np.ndarray) -> Optional[List[Tuple]]:
        """Simple trajectory generator which moves to an offset from 0th action,
        and then executes the given trajectory."""
        # grasp_pos, grasp_quat = to_pos_quat(grasp)
        self.robot.switch_to_manipulation_mode()
        trajectory = []

        # TODO: add skill-specific standoffs from 0th action
        joint_pos_pre = self.robot.manip.get_joint_positions()
        initial_pt = ("initial", joint_pos_pre, False)
        trajectory.append(initial_pt)

        num_actions = skill_actions.shape[0]
        for i in range(num_actions):
            desired_pos = skill_actions[i, 0:3]
            desired_quat = skill_actions[i, 3:7]
            desired_cfg, success, _ = self.robot.model.manip_ik(
                (desired_pos, desired_quat), q0=None
            )
            if success and desired_cfg is not None:
                desired_pt = (
                    f"action_{i}",
                    self.robot.model.config_to_manip_command(desired_cfg),
                    bool(skill_actions[i, 7]),
                )
                trajectory.append(desired_pt)
            else:
                print("-> could not solve for skill")
                return None
        trajectory.append(initial_pt)
        return trajectory

    def _send_action_to_tf(self, action):
        pass

    def try_executing_skill(
        self, action: np.ndarray, wait_for_input: bool = False
    ) -> bool:
        """Execute a predefined end-effector trajectory. Expected input is NUM_WAYPOINTSx8,
        where each waypoint is: pos(3-val), ori(4-val), gripper(1-val)"""
        # assert grasp.shape == (4, 4)
        # self._send_action_to_tf(grasp)

        # Generate a plan
        trajectory = self.plan_for_skill(action)

        if trajectory is None:
            print("Planning failed")
            return False

        for i, (name, waypoint, should_grasp) in enumerate(trajectory):
            self.robot.manip.goto_joint_positions(waypoint)
            # TODO: remove this delay - it's to make sure we don't start moving again too early
            rospy.sleep(0.1)
            # self._publish_current_ee_pose()
            if should_grasp:
                self.robot.manip.close_gripper()
            if wait_for_input:
                input(f"{i+1}) went to {name}")
            else:
                print(f"{i+1}) went to {name}")
        print(">>>--->> SKILL ATTEMPT COMPLETE <<---<<<")
        return True
