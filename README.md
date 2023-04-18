# Home Robot

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield&circle-token=282f21120e0b390d466913ef0c0a92f0048d52a3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/home-robot/blob/main/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Your open-source robotic mobile manipulation stack

## Core Concepts

This package assumes you have a low-cost mobile robot with limited compute -- initially a [Hello Robot Stretch](hello-robot.com/) - and a "workstation" with more GPU compute. Both are assumed to be running on the same network.

In general this is the recommended workflow for hardware robots:
  - Turn on your robot; for the Stretch, run `stretch_robot_home.py` to get it ready to use.
  - From your workstation, SSH into the robot and start a [ROS launch file](http://wiki.ros.org/roslaunch) which brings up necessary low-level control and hardware drivers.
  - If desired, run [rviz](http://wiki.ros.org/rviz) on the workstation to see what the robot is seeing.
  - Start running your AI code on the workstation!

We provide a couple connections for useful perception libraries like [Detic](https://github.com/facebookresearch/Detic) and [Contact Graspnet](https://github.com/NVlabs/contact_graspnet), which you can then use as a part of your methods.

## Installation & Usage

This project contains numerous packages. See individual package docs for corresponding details & instructions.

| Resource | Description |
| -------- | ----------- |
| [home_robot](src/home_robot) | Core package |
| [home_robot_hw](src/home_robot_hw) | ROS package containing hardware drivers for the Hello Stretch Robot |
| [home_robot_sim](src/home_robot_sim) | Simulation |
| [home_robot_client](src/home_robot_client) | Minimal remote client |

### Working with Stretch Environments
Assuming `roslaunch home_robot_hw startup_stretch_hector_slam.launch` is running in a separate terminal, you could run the following scripts:

Run simple navigation example (moves the robot forward by 0.25 m): 
  ```sh
  python src/home_robot_hw/home_robot_hw/env/simple_navigation_env.py
  ```
This file also serves as a simple example of how to setup your own environments implementing Stretch functionality. Every environment interfaces with the base Stretch controllers, models and environments to implement application-level requirements.

### Collecting data on the real robot
We provide scripts to collect data in H5 format using `Recorder` class. Follow the instructions for recording these files. If your application needs more/fewer data-sources, sub-class `Recorder` and over-ride the `save_frame` method
Collect the data through teleoperation.
  ```sh
  python collect_h5.py --task_name TASK_NAME  --dir_path DIR_PATH
  ```
  This will save the teleoperation files at `DIR_PATH/TASK_NAME-{iteration}/{datetime}.h5`.

  Turn on the controller and make sure it is connected to the robot (top two blue lights should be on). To give trajectory commands through the Xbox controller, open a separate terminal and:
  ```sh
  rosrun joy joy_node
  ```
Refer to [official hello robot keybindings](https://docs.hello-robot.com/0.2/stretch-tutorials/getting_started/images/xbox.png) to learn how to operate Stretch with the controller. We also provide a `Dataloaders` to load data into torch (WIP). 

### Syncing code between Robot and Workstation

Let `ROBOT_IP` store the robot's IP and let `WORKSTATION_IP` store the workstation's IP. If your local network doesn't have access to internet we recommend using `rsync` with `--update` flag to sync your code changes across the machines. Usage:
```
rsync -rvu /abs/path/to/local/home-robot $ROBOT_USER@$ROBOT_IP:/abs/path/to/remote/home-robot
```

The above command will do a *r*ecursive *u*pdating of changed files while echoing a *v*erbose output.


## Troubleshooting 

- `ImportError: cannot import name 'gcd' from 'fractions'`: Launch ros nodes from an env with Python 3.8 instead of 3.9


## Code Contribution

We enforce linters for our code. The `lint` test will not pass if your code does not conform.

Install the git [pre-commit](https://pre-commit.com/) hooks by running
```bash
python -m pip install pre-commit
cd $HOME_ROBOT_ROOT
pre-commit install
```

To format manually, run: `pre-commit run --show-diff-on-failure --all-files`

## License
Home Robot is MIT licensed. See the [LICENSE](./LICENSE) for details.

## References (temp)

- [hello-robot/stretch_body](https://github.com/hello-robot/stretch_body)
  - Base API for interacting with the Stretch robot
  - Some scripts for interacting with the Stretch
- [hello-robot/stretch_ros](https://github.com/hello-robot/stretch_ros)
  - Builds on top of stretch_body
  - ROS-related code for Stretch
- [RoboStack/ros-noetic](https://github.com/RoboStack/ros-noetic)
  - Conda stream with ROS binaries
