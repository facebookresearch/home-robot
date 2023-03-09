import click
import h5py
import rospy
import tf2_ros
from slap_manipulation.utils.h5_utils import plot_ee_pose, view_keyframe_imgs


@click.command()
@click.option(
    "--h5-file",
    default="",
    help="Absolute or relative path to h5 file to be introspected",
)
@click.option("--trial", default="", help="Trial name as a string which to introspect")
def main(h5_file, trial):
    if not h5_file:
        print("No file path provided. Program will exit")
        return
    rospy.init_node("h5_introspection")
    ros_pub = tf2_ros.TransformBroadcaster()
    file = h5py.File(h5_file, "r")
    view_keyframe_imgs(file, trial)
    plot_ee_pose(file, trial, ros_pub)


if __name__ == "__main__":
    main()
