# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import numpy
from home_robot.agent.motion.stretch import HelloStretch
from home_robot.agent.motion.stretch import STRETCH_HOME_Q, STRETCH_GRASP_OFFSET
from home_robot.utils.bullet import PbArticulatedObject
from home_robot.utils.pose import to_matrix, to_pos_quat


def fk_ik_helper(robot, q):
    """ do (1) generate random robot 
        generate fk for it; randomize robot
        """
    pass


def get_ik_solver():
    return HelloStretch(urdf_path='./assets/hab_stretch/urdf/', visualize=True)

def test_ik():
    """
    Goal pos and rot: (array([-0.10281811, -0.7189281 ,  0.71703106], dtype=float32), array([-0.7079143 ,  0.12421559,  0.1409881 , -0.68084526]))
    Current best solution: (array([-0.1350856 , -0.71864623,  0.71646219]), array([ 0.7084716 , -0.12145648, -0.13812223,  0.68135047]))

    2nd Goal pos and rot: (array([-0.01556295, -0.51387864,  0.8205258 ], dtype=float32), array([-0.7090214 ,  0.12297839,  0.14050716, -0.6800168 ]))
    Current best solution: (array([-0.12925884, -0.51288551,  0.8185215 ]), array([ 0.71091503, -0.1131743 , -0.13030495,  0.68177122]))
    """
    robot = get_ik_solver()
    q0 = STRETCH_HOME_Q
    block = PbArticulatedObject('red_block', './assets/red_block.urdf', client=robot.ref.client)
    robot.set_config(q0)
    test_poses = [
        ([-0.10281811, -0.7189281 ,  0.71703106], [-0.7079143 ,  0.12421559,  0.1409881 , -0.68084526]),
        ([-0.01556295, -0.51387864,  0.8205258 ], [-0.7090214 ,  0.12297839,  0.14050716, -0.6800168 ]),
        ]
    test_poses = [to_pos_quat(to_matrix(pos, quat) @ STRETCH_GRASP_OFFSET) for pos, quat in test_poses]
    test_poses = [robot.get_ee_pose()] + test_poses
    for pos, quat in test_poses:
        print("-------- 1 ---------")
        print("GOAL:", pos, quat)
        block.set_pose(pos, quat)
        res = robot.manip_ik((pos, quat), q0, relative=True)
        robot.set_config(res)
        pos2, quat2 = robot.get_ee_pose()
        print("PRED:", pos2, quat2)
        print("x motion:", res[0])
        input('press enter to continue')

        print("-------- 2 ---------")
        pos, quat = robot.get_ee_pose()
        print("GOAL:", pos, quat)
        block.set_pose(pos, quat)
        res = robot.manip_ik((pos, quat), q0, relative=True)
        robot.set_config(res)
        pos2, quat2 = robot.get_ee_pose()
        print("PRED:", pos2, quat2)
        print("x motion:", res[0])
        input('press enter to continue')
 
                  

def test_fk_ik():
    np.random.seed(0)
    rob = HelloStretch()
    for i in range(1000):
        res = fk_ik_helper(rob, q)


if __name__ == '__main__':
    test_ik()
