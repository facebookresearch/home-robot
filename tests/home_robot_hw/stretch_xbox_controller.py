# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import rospy

from home_robot.motion.stretch import HelloStretchKinematics
from home_robot_hw.ros.path import get_package_path
from home_robot_hw.ros.recorder import Recorder, pngs_to_mp4
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController

if __name__ == "__main__":
    rospy.init_node("xbox_controller")

    output_filename = "test_data.h5"
    video_filename = "test"
    fps = 10

    stretch_planner_urdf_path = os.path.join(
        get_package_path(), "../assets/hab_stretch/urdf/planner_calibrated.urdf"
    )
    model = HelloStretchKinematics(
        visualize=False,
        root="",
        urdf_path=stretch_planner_urdf_path,
    )
    # Create recorder - if we are going to use it
    recorder = Recorder(output_filename, model=model)
    controller = StretchXboxController(
        model, on_first_joystick_input=recorder.start_recording
    )

    recorder.spin(rate=fps)

    # Write to video (will trigger after a ctrl+C to stop the spin): TODO: trigger less hackily
    print("Writing video...")
    pngs_to_mp4(output_filename, "rgb", video_filename, fps=fps)
