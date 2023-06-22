# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

np.random.seed(0)


def generate_controller_input():
    yaw_on = np.random.choice([0.0, 1.0])
    loc_rand = np.random.randn(3).tolist()
    is_new_goal = np.random.choice([0.0, 1.0])
    goal_rand = np.random.randn(3).tolist()
    return yaw_on, loc_rand, is_new_goal, goal_rand


def get_controller_output(controller, input):
    yaw_on, loc_rand, is_new_goal, goal_rand = input

    controller.set_yaw_tracking(yaw_on)
    controller.update_pose_feedback(loc_rand)
    if is_new_goal:
        controller.update_goal(goal_rand)

    return controller.compute_control()
