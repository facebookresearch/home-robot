import click
import h5py
import rospy
import tf2_ros

from home_robot.utils.data_tools.h5_utils import plot_ee_pose, view_keyframe_imgs
from home_robot_hw.env.stretch_manipulation_env import StretchManipulationEnv


@click.command()
@click.option(
    "--h5-file",
    default="",
    help="Absolute or relative path to h5 file to be introspected",
)
@click.option(
    "--trial", default=0, type=int, help="Trial name as a string which to introspect"
)
@click.option("--replay", default=False, help="To replay the episode on Stretch")
def main(h5_file, trial, replay):
    if not h5_file:
        print("No file path provided. Program will exit")
        return
    rospy.init_node("h5_introspection")
    ros_pub = tf2_ros.TransformBroadcaster()
    file = h5py.File(h5_file, "r")
    trial = list(file.keys())[trial]
    view_keyframe_imgs(file, trial)
    print(f"Key schema: {file[trial].keys()}")
    ee_pose = plot_ee_pose(file, trial, ros_pub)
    if replay:
        robot = StretchManipulationEnv(init_cameras=True)
        # TODO: add gripper-action to StretchManipulationEnv.apply_action
        for pose in ee_pose:
            robot.apply_action({"pos": pose[0], "rot": pose[1]})


if __name__ == "__main__":
    main()
