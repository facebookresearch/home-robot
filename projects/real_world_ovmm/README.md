# Stand-alone grasping setup

## Setup

1. Follow instructions in [home_robot_hw](../../src/home_robot_hw/README.md) to set up the Home Robot hardware stack on both your Workstation and Robot.
2. Install additional dependencies for OVMM
    ```sh
    pip install -r projects/real_world_ovmm/requirements.txt
    ```
3. Install habitat-lab dependencies
    ```sh
    git submodule update --init --recursive src/third_party/habitat-lab
    pip install -e habitat-lab
    pip install -e src/third_party/habitat-lab/habitat-lab
    pip install -e src/third_party/habitat-lab/habitat-baselines
    ```

## Usage

### Launch on robot
```sh
roslaunch home_robot startup_stretch_hector_slam.launch
```

To make debugging easier, it is recommended to run
```sh
roscore
```
in a separate terminal.

### Launch on server
```sh
# Launch grasping server
python -m home_robot_hw.nodes.simple_grasp_server

# Optional 1: Launch rviz 
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
# Optional 2: Launch rviz from a ROS environment with home_robot_hw installed
roslaunch home_robot visualization.launch

# Run stand-alone pick & place script
python projects/real_world_ovmm/eval_episode.py
```

## Troubleshooting the robot

Reset the robot hardware after some component fails:
1. Kill all running ROS nodes
2. `stretch_robot_home.py`

Check if the robot hardware is functional: `stretch_robot_system_check.py`