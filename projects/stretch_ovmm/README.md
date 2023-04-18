# Stand-alone grasping setup

## Setup

1. Follow instructions in [home_robot_hw](../../src/home_robot_hw/README.md) to set up the Home Robot hardware stack on both your Workstation and Robot.
2. Install additional dependencies for OVMM
    ```sh
    pip install -r projects/stretch_ovmm/requirements.txt
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

# Optional: Launch rviz
roslaunch home_robot visualization.launch

# Run stand-alone grasping script
python projects/stretch_ovmm/eval_episode.py
```

## Troubleshooting the robot
```sh
# To control the robot to a starting position
roslaunch home_robot controller.launch

# To reset everything after some component fails
stretch_robot_home.py
```
