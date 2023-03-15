# Home Robot

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield&circle-token=282f21120e0b390d466913ef0c0a92f0048d52a3)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/home-robot/blob/main/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Mostly Hello Stretch infrastructure

## Core Concepts

This package assumes you have a low-cost mobile robot with limited compute -- initially a [Hello Robot Stretch](hello-robot.com/) - and a "workstation" with more GPU compute. Both are assumed to be running on the same network.

In general this is the recommended workflow:
  - Turn on your robot; for the Stretch, run `stretch_robot_home.py` to get it ready to use.
  - From your workstation, connect to the robot and start a [ROS launch file](http://wiki.ros.org/roslaunch) which brings up necessary low-level control and hardware drivers.
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

### Getting Started on the Hello Stretch

1. Clone the repo on your Stretch Robot and the local GPU machine.
    ```sh
    git clone https://github.com/facebookresearch/home-robot.git
    
    # Make sure you update all submodules by running
    git submodule update --recursive --init
    ```
    If the modules do not update as expected, make sure that you have added the [SSH public key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) of your robot and machine to authenticate your Github account.
    
    Set the env variables
    ```sh
    export HOME_ROBOT_ROOT=$(pwd)/home-robot
    ```

1. Install the core [home_robot](src/home_robot) python package. Main aspects reproduced here for convenience:

    For installing on workstation-side:
    ```sh
    cd $HOME_ROBOT_ROOT/src/home_robot
    mamba env create -n home_robot -f environment.yml
    conda activate home_robot
    pip install -e .
    ```

    For installing on robot-side:
    ```sh
    cd $HOME_ROBOT_ROOT/src/home_robot
    pip install -e .
    ```
1. Install [home_robot_hw](src/home_robot_hw/install.md) and complete the setup. Main aspects reproduced here for convenience:
    ```sh
    # Create symlink in catkin workspace
    ln -s /abs/path/to/home-robot/src/home_robot_hw $HOME/catkin_ws/src/home_robot_hw

    # Install dependencies for catkin
    pip install empy catkin_pkg rospkg

    # Build catkin workspace
    cd ~/catkin_ws  
    rm -rf build/ devel/  # Optional to ignore stale cached files
    catkin_make

    # Add newly built setup.bash to .bashrc
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    ```

 
1. Launch the ROS hardware stack:
    ```sh
    conda deactivate  # If you are using conda - not required on robot!
    roslaunch home_robot_hw startup_stretch_hector_slam.launch
    ```
    Sanity check: run hardware test to test head movement, navigation (moved forward-left, turned to face right) and manipulation (lift upwards and extend outwards by 20cm).  
    ```sh 
    python tests/hw_manual_test.py
    ```  

### Development

To develop in `home-robot`, install the git pre-commit hooks:
```
python -m pip install pre-commit
cd $HOME_ROBOT_ROOT
pre-commit install
```

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

### Launching Grasping Demo (outdated)

You need to create a catkin workspace on your workstation in order to run this demo, as this is where we will run [Contact Graspnet](https://github.com/cpaxton/contact_graspnet/tree/cpaxton/devel).

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

### Syncing code between Robot and Workstation

Let `ROBOT_IP` store the robot's IP and let `WORKSTATION_IP` store the workstation's IP. If your local network doesn't have access to internet we recommend using `rsync` with `--update` flag to sync your code changes across the machines. Usage:
```
rsync -rvu /abs/path/to/local/home-robot $ROBOT_USER@$ROBOT_IP:/abs/path/to/remote/home-robot
```

The above command will do a *r*ecursive *u*pdating of changed files while echoing a *v*erbose output.

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

