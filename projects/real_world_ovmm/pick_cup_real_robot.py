# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import click
import rospy

from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


@click.command()
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--visualize-masks", default=False, is_flag=True)
@click.option("--visualize-planner", default=False, is_flag=True)
def main(dry_run, visualize_masks, visualize_planner):
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")

    env = StretchPickandPlaceEnv(visualize_planner=visualize_planner)
    env.reset(goal="cup")
    # TODO Call apply_action() instead of try_grasping()
    env.try_grasping(visualize_masks=visualize_masks, dry_run=dry_run)


if __name__ == "__main__":
    main()
