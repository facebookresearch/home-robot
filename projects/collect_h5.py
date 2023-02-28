import os
from datetime import datetime

import click
import rospy

from home_robot.agent.motion.stretch import HelloStretch
from home_robot_hw.ros.path import get_urdf_dir
from home_robot_hw.ros.recorder import Recorder
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController


class EpisodeManager(object):
    """Episode manager class for creating, closing and managing episodic data
    This works in a task-centric fashion right now. Creating or using the
    task-directory inside ~/H5s/ and adding new files with today's date
    and time.
    """

    def __init__(self, task_name, dir_path):
        stretch_planner_urdf_path = get_urdf_dir()
        self.model = HelloStretch(
            visualize=False,
            root="",
            urdf_path=stretch_planner_urdf_path,
        )
        self.task_name = task_name
        self.file_path = os.path.join(dir_path, task_name)
        os.makedirs(self.file_path, exist_ok=True)
        self.controller = StretchXboxController(
            self.model,
            start_button_callback=self.toggle_episode,
            back_button_callback=self.record_keyframe,
        )
        self._is_recording = False

    def toggle_episode(self):
        if not self._is_recording:
            self._is_recording = True
            date_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")
            filename = os.path.join(self.file_path, date_time_string)
            self._keyframe_recorder = Recorder(filename)
            self._keyframe_recorder.start_recording(self.task_name)
        else:
            self._is_recording = False
            self._keyframe_recorder.save_frame()
            self._keyframe_recorder.finish_recording()

    def record_keyframe(self):
        """add a keyframe to the current episode"""
        self._keyframe_recorder.save_frame()


@click.command()
@click.option("--task-name", default="task", help="Name of the task to record")
def main(task_name):
    rospy.init_node("h5_demo_recorder")
    rate = rospy.Rate(10)
    data_directory = os.path.expanduser("~/H5s/")
    em = EpisodeManager(task_name, data_directory)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
    print("Shutting down h5 demo recording script")


if __name__ == "__main__":
    main()
