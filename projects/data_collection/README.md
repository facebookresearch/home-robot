# Data Collection

## Collect H5 Files

1. Make sure the main hector-slam launch file is up and running.
1. On a new terminal run `rosrun joy joy_node`
1. Turn on the controller and make sure it is connect to the PC
1. Run the data collection script: `python projects/data_collection/collect_h5.py --task-name <name of task> --dir-path <name of dir path>`.

Note that the script collect keyframe based demonstrations at the moment, i.e. user needs to communicate episode start, moments of interest during episode and episode end to the script.

### Controls for Data Collection

- To start an episode press the `Start` button on the controller (make sure the script starts writing to a file)
- To communicate keyframes, press `Back` button
- To end the episode press `Start` button again (we collect another frame at the end of the episode, it does not need explicit marking as a keyframe)

### About the Data

This collects [Trial](src/home_robot/home_robot/utils/data_tools/loader.py) containing these keys:
- temporal:  joint position `q`, joint velocity `dq`, end effector pose `ee_pose`, `base_pose`, `camera_pose`, id of the image frame (when user pressed start) `user_keyframe`.
- config: 
- image: `rgb` and `depth`
To modify and record your own data, inherit [Recorder](src/home_robot_hw/home_robot_hw/ros/recorder.py) and use `add_frame` for temporal keys, `add_img_frame` for image keys and `add_config` for config keys. 


## Load the collected H5 data files with torch Dataloader

To test the dataloader, rename `03-06_13-19-05.h5.small` to `03-06_13-19-05.h5`.

`python projects/data_collection/tutorial_h5_dataloader.py`

