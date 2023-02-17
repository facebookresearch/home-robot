# Home Robot

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield&circle-token=282f21120e0b390d466913ef0c0a92f0048d52a3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mostly Hello Stretch infrastructure

## Installation & Usage

This project contains numerous packages. See individual package docs for corresponding details & instructions.

| Resource | Description |
| -------- | ----------- |
| [home_robot](src/home_robot) | Core package |
| [home_robot_hw](src/home_robot_hw) | ROS package containing hardware drivers for the Hello Stretch Robot |
| [home_robot_sim](src/home_robot_sim) | Simulation |
| [home_robot_client](src/home_robot_client) | Minimal remote client |

### Getting Started on the Hello Stretch

1. SSH into the onboard computer on the Hello Stretch.
1. Install [home_robot_hw](src/home_robot_hw/install.md).
1. Install [home_robot](src/home_robot).
1. Launch the ROS hardware stack:
    ```sh
    conda deactivate
    roslaunch home_robot startup_stretch_hector_slam.launch
    ```
1. In a separate shell, launch home-robot helper nodes:
    ```sh
    conda activate home_robot
    python -m home_robot.nodes.state_estimator &
    python -m home_robot.nodes.goto_controller &
    ```
1. Launch interactive client: `python -m home_robot.client.local_hello_robot`

You should then be able to command the robot using the following commands:
```py
# Query states
robot.get_base_state()  # returns base location in the form of [x, y, rz]

# Mode switching
robot.switch_to_velocity_mode()  # enables base velocity control
robot.switch_to_navigation_mode()  # enables continuous navigation
robot.switch_to_manipulation_mode()  # enables gripper control

# Velocity mode
robot.set_velocity(v: float, w: float)  # directly sets the linear and angular velocity of robot base

# Navigation mode
robot.navigate_to(xyt: list, relative: bool = False, position_only: bool = False)

# Manipulation mode
robot.set_arm_joint_positions(joint_positions: list)  # joint positions: [BASE_TRANSLATION, ARM_LIFT, ARM_EXTENTION, WRIST_YAW, WRIST_PITCH, WRIST_ROLL]
robot.set_ee_pose(pos: list, quat: list, relative: bool = False)
```

Basic example:
```py
robot.get_base_state()  # Shows the robot's SE2 coordinates (should be [0, 0, 0])

robot.switch_to_navigation_mode()
robot.navigate_to([0.3, 0.3, 0.0])  # Sets SE2 target
robot.get_base_state()  # Shows the robot's SE2 coordinates (should be close to [0.3, 0.3, 0])

robot.switch_to_manipulation_mode()
robot.set_ee_pose([0.5, 0.6, 0.5], [0, 0, 0, 1])
```

### Launching Grasping Demo (outdated)

You need to create a catkin workspace on your server in order to run this demo, as this is where we will run [Contact Graspnet](https://github.com/cpaxton/contact_graspnet/tree/cpaxton/devel).

Contact graspnet is downloaded under `third_party/`, but there is a `CATKIN_IGNORE` file in this directory. You want to symlink this file out into your workspace:
```
ROSWS=/path/to/ros_ws
ln -s `rospack find home_robot`/third_party/contact_graspnet $ROSWS/src/contact_graspnet
```
... but it actually shouldn't be necessary. What is necessary is to build the grasp service defined in `home_robot` by placing it into `$ROSWS`.


Put the robot in its initial position, e.g. so the arm is facing cups you can pick up. On the robot side:
```
roslaunch home_robot startup_stretch_hector_slam.launch
```

#### Note: Contact GraspNet

Contact graspnet is supported as a way of generating candidate grasps for the Stretch to use on various objects. We have our own fork of [Contact Graspnet](https://github.com/cpaxton/contact_graspnet/tree/cpaxton/devel) which has been modified with a ROS interface.

Follow the installation instructions as normal and start it with:
```
conda activate contact_graspnet_env
~/src/contact_graspnet$ python contact_graspnet/graspnet_ros_server.py  --local_regions --filter_grasps
```

### Troubleshooting 

- `ImportError: cannot import name 'gcd' from 'fractions'`: Launch ros nodes from an env with Python 3.8 instead of 3.9


## Code Contribution

We enforce linters for our code. The `lint` test will not pass if your code does not conform.

Install the git [pre-commit](https://pre-commit.com/) hooks by running
  ```bash
  pip install pre-commit
  pre-commit install
  ```

To format manually, run: `pre-commit run --show-diff-on-failure --all-files`

## License
Home Robot is MIT licensed. See the [LICENSE](./LICENSE) for details.

## References (temp)

- [cpaxton/home_robot](https://github.com/cpaxton/home_robot)
  - Chris' repo for controlling stretch
- [facebookresearch/fairo](https://github.com/facebookresearch/fairo)
  - Robotics platform with a bunch of different stuff
  - [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis): Contains Torchscript controllers useful for exposing low-level control logic to the user side.
  - [Meta Robotics Platform(MRP)](https://github.com/facebookresearch/fairo/tree/main/mrp): Useful for launching & managing multiple processes within their own sandboxes (to prevent dependency conflicts).
  - The [perception](https://github.com/facebookresearch/fairo/tree/main/perception) folder contains a bunch of perception related modules
    - Polygrasp: A grasping library that uses GraspNet to generate grasps and Polymetis to execute them.
    - iphone_reader: iPhone slam module.
    - realsense_driver: A thin realsense wrapper
  - [droidlet/lowlevel/hello_robot](https://github.com/facebookresearch/fairo/tree/main/droidlet/lowlevel/hello_robot)
    - Austin's branch with the continuous navigation stuff: austinw/hello_goto_odom
    - Chris & Theo's branch with the grasping stuff: cpaxton/grasping-with-semantic-slam
    - [Nearest common ancester of all actively developing branches](https://github.com/facebookresearch/fairo/tree/c39ec9b99115596a11cb1af93a31f1045f92775e): Should migrate this snapshot into home-robot then work from there.
- [hello-robot/stretch_body](https://github.com/hello-robot/stretch_body)
  - Base API for interacting with the Stretch robot
  - Some scripts for interacting with the Stretch
- [hello-robot/stretch_firmware](https://github.com/hello-robot/stretch_firmware)
  - Arduino firmware for the Stretch
- [hello-robot/stretch_ros](https://github.com/hello-robot/stretch_ros)
  - Builds on top of stretch_body
  - ROS-related code for Stretch
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_ros2)
  - Development branch for ROS2
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_web_interface)
  - Web interface for teleoping Stretch
- [RoboStack/ros-noetic](https://github.com/RoboStack/ros-noetic)
  - Conda stream with ROS binaries
- [codekansas/strech-robot](https://github.com/codekansas/stretch-robot)
  - Some misc code for interacting with RealSense camera, streaming

