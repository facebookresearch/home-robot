# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import numpy as np

from home_robot.mapping.voxel.voxel import SparseVoxelMap
from home_robot.motion import ConfigurationSpace, Planner, PlanResult
from home_robot.utils.visualization import get_x_and_y_from_path


def plan_to_frontier(
    start: np.ndarray,
    planner: Planner,
    space: ConfigurationSpace,
    voxel_map: SparseVoxelMap,
    visualize: bool = False,
    try_to_plan_iter: int = 10,
    debug: bool = False,
    verbose: bool = False,
) -> PlanResult:
    """Simple helper function for planning to the frontier during exploration.

    Args:
        start(np.ndarray): len=3 array containing world x, y, and theta
        planner(Planner): what we will use to generate motion plans to frontier points
    """
    # extract goal using fmm planner
    tries = 0
    failed = False
    res = None
    start_is_valid = space.is_valid(start)
    print("\n----------- Planning to frontier -----------")
    print("Start is valid:", start_is_valid)
    if not start_is_valid:
        return PlanResult(False, reason="invalid start state")
    for goal in space.sample_closest_frontier(start, verbose=verbose, debug=debug):
        if goal is None:
            failed = True
            break
        goal = goal.cpu().numpy()
        print("       Start:", start)
        print("Sampled Goal:", goal)
        show_goal = np.zeros(3)
        show_goal[:2] = goal[:2]
        goal_is_valid = space.is_valid(goal)
        print("Start is valid:", start_is_valid)
        print(" Goal is valid:", goal_is_valid)
        if not goal_is_valid:
            print(" -> resample goal.")
            continue
        # plan to the sampled goal
        res = planner.plan(start, goal)
        print("Found plan:", res.success)
        if visualize:
            obstacles, explored = voxel_map.get_2d_map()
            img = (10 * obstacles) + explored
            space.draw_state_on_grid(img, start, weight=5)
            space.draw_state_on_grid(img, goal, weight=5)
            plt.imshow(img)
            if res.success:
                path = voxel_map.plan_to_grid_coords(res)
                x, y = get_x_and_y_from_path(path)
                plt.plot(y, x)
                plt.show()
        if res.success:
            break
        else:
            if visualize:
                plt.show()
            tries += 1
            if tries >= try_to_plan_iter:
                failed = True
                break
            continue
    else:
        print(" ------ no valid goals found!")
        failed = True
    if failed:
        print(" ------ sampling and planning failed! Might be no where left to go.")
        return PlanResult(False, reason="planning to frontier failed")
    return res
