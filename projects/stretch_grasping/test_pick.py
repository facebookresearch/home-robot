#!/usr/bin/env python
import numpy as np
import rospy

from home_robot.agent.hierarchical.pick_and_place_agent import PickAndPlaceAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


def run_experiment():

    config_path = "projects/stretch_grasping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(config=config)
    env.reset("table", "cup", "chair")

    pose = np.array(
        [
            [0.23301425, -0.97144842, -0.04463536, -0.00326367],
            [-0.97188458, -0.23103087, -0.04544342, -0.44448592],
            [0.0338338, 0.05396939, -0.99796923, 0.99206106],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )
    pos, quat = to_pos_quat(pose)
    pose1 = env.robot.head.get_pose()
    pose2 = env.robot.head.get_pose_in_base_coords()
    breakpoint()

    env.switch_to_manipulation_mode()
    env.grasp_planner.go_to_manip_mode()
    env.grasp_planner.try_executing_grasp(pose, wait_for_input=True)


if __name__ == "__main__":
    run_experiment()
