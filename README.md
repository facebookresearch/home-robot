# HomeRobot

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/home-robot/blob/main/LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/home-robot/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/home-robot/tree/main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://timothycrosley.github.io/isort/)

Your open-source robotic mobile manipulation stack!

HomeRobot lets you get started running a range of robotics tasks on a low-cost mobile manipulator, starting with _Open Vocabulary Mobile Manipulation_, or OVMM. OVMM is a challenging task which means that, in an unknown environment, a robot must:
  - Explore its environment
  - Find an object
  - Find a receptacle -- a location on which it must place this object
  - Put the object down on the receptacle.

## Core Concepts

This package assumes you have a low-cost mobile robot with limited compute -- initially a [Hello Robot Stretch](hello-robot.com/) -- and a "workstation" with more GPU compute. Both are assumed to be running on the same network.

This is the recommended workflow for hardware robots:
  - Turn on your robot; for the Stretch, run `stretch_robot_home.py` to get it ready to use.
  - From your workstation, SSH into the robot and start a [ROS launch file](http://wiki.ros.org/roslaunch) which brings up necessary low-level control and hardware drivers.
  - If desired, run [rviz](http://wiki.ros.org/rviz) on the workstation to see what the robot is seeing.
  - Start running your AI code on the workstation - For example, you can run `python projects/stretch_grasping/eval_episode.py` to run the OVMM task.

We provide a couple connections for useful perception libraries like [Detic](https://github.com/facebookresearch/Detic) and [Contact Graspnet](https://github.com/NVlabs/contact_graspnet), which you can then use as a part of your methods.

## Installation

### Preliminary

Installation on a workstation requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html).

Installation on a robot assumes Ubuntu 20.04 and [ROS Noetic](http://wiki.ros.org/noetic).

To set up the hardware stack on a Hello Robot  Stretch, see the [ROS installation instructions](src/home_robot_hw/install_robot.md) in `home_robot_hw`.

Proper network setup is crucial to getting good performance with HomeRobot. Low-cost mobile robots often do not have sufficient GPU to run state-of-the-art perception models. Instead, we rely on a client-server architecture, where ROS and low-level controllers run on the robot, and CPU- and GPU-intensive AI code runs on a workstation.

After following the installation instructions, we recommend setting up your `~/.bashrc` on the robot workstation:

```
# Whatever your workstation's IP address is
export WORKSTATION_IP=10.0.0.2
# Whatever your robot's IP address is
export HELLO_ROBOT_IP=10.0.0.6

# Path to the codebase
export HOME_ROBOT_ROOT=/path/to/home-robot

export ROS_IP=$WORKSTATION_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"

# Helpful alias - connect to the robot
alias ssh-robot="ssh hello-robot@$HELLO_ROBOT_IP"
```

On the robot side, start up the controllers with:
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

### Workstation Instructions

To set up your workstation, follow these instructions:

#### 1. Create Your Environment
```
# Create a conda env - use the version in home_robot_hw if you want to run on the robot
# Otherwise, you can use the version in src/home_robot
mamba env create -n home-robot -f src/home_robot_hw/environment.yml
conda activate home-robot
```

This should install pytorch; if you run into trouble, you may need to edit the installation to make sure you have the right CUDA version. See the [pytorch install notes](docs/install_pytorch.md) for more.

#### 2. Install Home Robot Packages
```
# Install the core home_robot package
pip install -e src/home_robot

# Install home_robot_hw
pip install -e src/home_robot_hw
```

_Testing Real Robot Setup:_ Now you can run a couple commands to test your connection. If the `roscore` and the robot controllers are running properly, you can run `rostopic list` and should see a list of topics - streams of information coming from the robot. You can then run RVIZ to visualize the robot sensor output:

```
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
```

#### 3. Hardware Testing

Run the hardware manual test to make sure you can control the robot remotely. Ensure the robot has one meter of free space before running the script.

```
python tests/hw_manual_test.py
```

Follow the on-screen instructions. The robot should move through a set of configurations.


#### 4. Install Detic

Install [detectron2](https://detectron2.readthedocs.io/tutorials/install.html):
```
pip install -e src/third_party/detectron2
pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
```

Download Detic checkpoint as per the instructions [on the Detic github page](https://github.com/facebookresearch/Detic):
```bash
cd $HOME-ROBOT-PATH/src/home_robot/perception/detection/detic/Detic/
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate
```

You should be able to run the Detic demo script as per the Detic instructions to verify your installation was correct:
```bash
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

#### 5. Run Open Vocabulary Mobile Manipulation on Stretch

You should then be able to run the Stretch OVMM example.

Run a grasping server; either Contact Graspnet or our simple grasp server.
```
# For contact graspnet
cd $HOME_ROBOT_ROOT/src/third_party/contact_graspnet
conda activate contact_graspnet_env
python contact_graspnet/graspnet_ros_server.py  --local_regions --filter_grasps

# For simple grasping server
cd $HOME_ROBOT_ROOT
conda activate home-robot
python src/home_robot_hw/home_robot_hw/nodes/simple_grasp_server.py
```

Then you can run the OVMM example script:
```
cd $HOME_ROBOT_ROOT
python projects/stretch_ovmm/eval_episode.py
```

#### 6. Simulation Setup

To set up the simulation stack with Habitat, see the [installation instructions](src/home_robot_sim/README.md) in `home_robot_sim`. You first need to install AI habitat and the simulation package:
```
# Install habitat sim and update submodules
mamba env update -f src/home_robot_sim/environment.yml

# Install habitat lab on the correct (object rearrange) branch
git submodule update --init --recursive
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines

# Install home robot sim interfaces
pip install -e src/home_robot_sim
```

And then download the assets as described in the [installation instructions](src/home_robot_sim/README.md#Ddataset-setup).

To test your installation, you can run:
```
python projects/stretch_ovmm/eval_vectorized.py
```

For more details on the OVMM challenge, see the [Habitat OVMM readme](projects/stretch_ovmm/README.md).


## Code Contribution

We welcome contributions to HomeRobot.

There are two main classes in HomeRobot that you need to be concerned with:
  - *Environments* extend the [abstract Environment class](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/abstract_env.py) and provide *observations* of the world, and a way to *apply actions*.
  - *Agents* extend the [abstract Agent class](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/abstract_agent.py), which takes in an [observation](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/interfaces.py#L95) and produces an [action](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/interfaces.py#L50).

Generally, new methods will be implemented as Agents.

### Organization

[HomeRobot](https://github.com/facebookresearch/home-robot/) is broken up into three different packages:

| Resource | Description |
| -------- | ----------- |
| [home_robot](src/home_robot) | Core package containing agents and interfaces |
| [home_robot_sim](src/home_robot_sim) | OVMM simulation environment based on [AI Habitat](https://aihabitat.org/) |
| [home_robot_hw](src/home_robot_hw) | ROS package containing hardware interfaces for the Hello Robot Stretch |

The [home_robot](src/home_robot) package contains embodiment-agnostic agent code, such as our [ObjectNav agent](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent.py) (finds objects in scenes) and our [hierarchical OVMM agent](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/agent/ovmm_agent/ovmm_agent.py). YThese agents can be extended or modified to implement your own solution.

Importantly, agents use a fixed set of [interfaces](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/interfaces.py) which are overridden to provide access to 

The [home_robot_sim](src/home_robot_sim) package contains code for interface

### Style

We use linters for enforcing good code style. The `lint` test will not pass if your code does not conform.

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
