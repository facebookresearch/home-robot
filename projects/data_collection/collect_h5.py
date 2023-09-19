# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from datetime import datetime

import click
import rospy

from home_robot.motion.stretch import HelloStretchKinematics
from home_robot_hw.ros.path import get_urdf_dir
from home_robot_hw.ros.recorder import Recorder
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController


class EpisodeManager(object):
    """Episode manager class for creating, closing and managing episodic data
    This works in a task-centric fashion right now. Creating or using the
    task-directory inside ~/H5s/ and adding new files with today's date
    and time.
    """

    def __init__(self, task_name: str, dir_path: str = "./H5s/"):
        """
        task_name: name of the task to record, string
        dir_path: path to the directory where the task directory will be created

        The resulting directory structure is as follows:
            dir_path
            |--task_name_0
            |  |--date_time_string_0.h5
            |  |--date_time_string_1.h5
            |--task_name_1
            |  |--date_time_string_0.h5
            ...
        A new date_time_string file is created whenever this script is run for a task
        """
        stretch_planner_urdf_path = get_urdf_dir()
        self.model = HelloStretchKinematics(
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
        date_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        filename = os.path.join(self.file_path, date_time_string + ".h5")
        self._keyframe_recorder = Recorder(filename)
        self._k_idx = 0
        self._episode_count = 0
        self._success_count = 0
        self._failed_count = 0

    def toggle_episode(self):
        """toggles episode recording state
        Note: This script automatically collects a keyframe at the beginning when
        episode is started. Every frame to be considered for task should be explicitly marked as keyframe
        """
        if not self._is_recording:
            self._is_recording = True
            self._keyframe_recorder.start_recording(self.task_name)
            self._keyframe_recorder.save_frame()
            print("Start frame saved")
        else:
            self._is_recording = False
            status = self._keyframe_recorder.finish_recording()
            self._episode_count += 1
            if status == 1:
                self._success_count += 1
            else:
                self._failed_count += 1
            self._k_idx = 0
            print(
                f"Total trials: {self._episode_count}, Failed: {self._failed_count}, Succeeded: {self._success_count}"
            )

    def record_keyframe(self):
        """add a keyframe to the current episode"""
        self._keyframe_recorder.save_frame(is_keyframe=True)
        print(f"Keyframe saved: {self._k_idx}")
        self._k_idx += 1


@click.command()
@click.option("--task-name", default="task", help="Name of the task to record")
@click.option("--dir-path", default="./H5s/", help="Path of root data directory")
def main(task_name, dir_path):
    em = EpisodeManager(task_name, dir_path)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
    print("Shutting down h5 demo recording script")


if __name__ == "__main__":
    main()
