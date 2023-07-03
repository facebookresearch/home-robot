![](docs/HomeRobot_Logo_Horiz_Color_white_bg.png)

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

## 🏠🤖 Challenge  🚀
The objective of the HomeRobot: OVMM Challenge is to create a platform that enables researchers to develop agents that can navigate unfamiliar environments, manipulate novel objects, and move away from closed object classes towards open-vocabulary natural language. This challenge aims to facilitate cross-cutting research in embodied AI using recent advances in machine learning, computer vision, natural language, and robotics.

Check out the [Neurips 2023 HomeRobot Open-Vocabulary Mobile Manipulation Challenge!](https://aihabitat.org/challenge/2023_homerobot_ovmm/)

### Participation Guidelines

Participate in the contest by registering on the EvalAI challenge page (link coming soon!) and creating a team. Participants will upload docker containers with their agents that are evaluated on an AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to ensure it is working. Instructions for training, local evaluation, and online submission are provided below.

#### Prerequisites 
Make sure you have [Docker](https://docs.docker.com/engine/install/ubuntu/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. 

Optionally, you can [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) if you don’t want to preface the `docker` command with `sudo`.

#### Local Evaluation

1. Clone the challenge repository:
   ```
   git clone https://github.com/facebookresearch/home-robot.git
   ```
1. Navigate to `projects/habitat_ovmm`
   ```
   cd projects/habitat_ovmm
   ```
1. Implement your own agent or try our baseline agent, located in [projects/habitat_ovmm/eval_baselines_agent.py](projects/habitat_ovmm/eval_baselines_agent.py). 
1. Modify the provided [projects/habitat_ovmm/docker/ovmm_baseline.Dockerfile](projects/habitat_ovmm/docker/ovmm_baseline.Dockerfile) if you need custom modifications. Let’s say your code needs `<some extra package>`, this dependency should be pip installed inside a conda environment called `home-robot` that is shipped with our HomeRobot challenge docker, as shown below:
    ```dockerfile
    FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023-dev2

    # install dependencies in the home-robot conda environment
    RUN /bin/bash -c ". activate home-robot; pip install <some extra package>"

    ADD eval_baselines_agent.py agent.py
    ADD submission.sh submission.sh

    CMD ["/bin/bash", "-c", ". activate home-robot; export PYTHONPATH=/home-robot/projects/habitat_ovmm:$PYTHONPATH; bash submission.sh"]
    ```
1. Build your Docker image using:

    ```
    docker build . -f docker/ovmm_baseline.Dockerfile -t ovmm_baseline_submission
    ```
     
    *Note:* Please, make sure that you keep your local version of `fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023-dev2` image up to date with the image we have hosted on [dockerhub](https://hub.docker.com/r/fairembodied/habitat-challenge/tags). This can be done by pruning all cached images, using:
    ```
    docker system prune -a
    ```
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to agent in `agent.py`

1. Download all the required data into the `home-robot/data` directory (see [Habitat OVMM readme](projects/habitat_ovmm/README.md)). Then in your `docker run` command mount `home-robot/data` data folder to the `home-robot/data` folder in the Docker image (see `./scripts/test_local.sh` for reference).
     
1. Evaluate your docker container locally:
    ```bash
    ./scripts/test_local.sh --docker-name ovmm_baseline_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    Arguments:
    {
        "habitat_config_path": "ovmm/ovmm_eval.yaml",
        "baseline_config_path": "projects/habitat_ovmm/configs/agent/hssd_eval.yaml",
        "opts": []
    }
    ----------------------------------------------------------------------------------------------------
    Configs:

    ----------------------------------------------------------------------------------------------------
    pybullet build time: May 20 2022 19:45:31
    2023-07-03 15:04:05,629 Initializing dataset OVMMDataset-v0
    2023-07-03 15:04:06,094 initializing sim OVMMSim-v0
    2023-07-03 15:04:08,686 Initializing task OVMMNavToObjTask-v0
    Running eval on [1200] episodes
    Initializing episode...
    [OVMM AGENT] step heuristic nav policy
    Executing skill NAV_TO_OBJ at timestep 1
    [OVMM AGENT] step heuristic nav policy
    Executing skill NAV_TO_OBJ at timestep 2
    [OVMM AGENT] step heuristic nav policy
    Executing skill NAV_TO_OBJ at timestep 3
    [OVMM AGENT] step heuristic nav policy
    Executing skill NAV_TO_OBJ at timestep 4
    [OVMM AGENT] step heuristic nav policy
    Executing skill NAV_TO_OBJ at timestep 5
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.** 

#### Online Submission

Follow instructions in the `submit` tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI `>= 1.2.3`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.5"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push ovmm_baseline_submission --phase <phase-name>
```

The challenge consists of the following phases:

1. **Minival phase**: The purpose of this phase is sanity checking — to confirm that remote evaluation reports the same result as local evaluation. Each team is allowed up to 100 submissions per day. We will disqualify teams that spam the servers.
1. **Test standard phase**: The purpose of this phase/split is to serve as the public leaderboard establishing the state of the art. This is what should be used to report results in papers. Each team is allowed up to 10 submissions per day, to be used judiciously.
1. **Test challenge phase**: This split will be used to decide challenge teams who will proceed to Stage 2 Evaluation. Each team is allowed a total of 5 submissions until the end of challenge submission phase. The highest performing of these 5 will be automatically chosen.

Simulation agents will be evaluated on an AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. Agents will be evaluated on 1000 episodes and will have a total available time of 48 hours to finish each run. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them by opening an issue on this repository.

### DD-PPO Training Starter Code
Please refer to the Training DD-PPO skills section of the [Habitat OVMM readme](projects/habitat_ovmm/README.md#training-dd-ppo-skills) for more details.


## Core Concepts

This package assumes you have a low-cost mobile robot with limited compute -- initially a [Hello Robot Stretch](https://hello-robot.com/stretch-2) -- and a "workstation" with more GPU compute. Both are assumed to be running on the same network.

This is the recommended workflow for hardware robots:
  - Turn on your robot; for the Stretch, run `stretch_robot_home.py` to get it ready to use.
  - From your workstation, SSH into the robot and start a [ROS launch file](http://wiki.ros.org/roslaunch) which brings up necessary low-level control and hardware drivers.
  - If desired, run [rviz](http://wiki.ros.org/rviz) on the workstation to see what the robot is seeing.
  - Start running your AI code on the workstation - For example, you can run `python projects/stretch_grasping/eval_episode.py` to run the OVMM task.

We provide a couple connections for useful perception libraries like [Detic](https://github.com/facebookresearch/Detic) and [Contact Graspnet](https://github.com/NVlabs/contact_graspnet), which you can then use as a part of your methods.

## Installation

### Preliminary

HomeRobot requires Python 3.9. Installation on a workstation requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html). Installation on a robot assumes Ubuntu 20.04 and [ROS Noetic](http://wiki.ros.org/noetic).

To set up the hardware stack on a Hello Robot  Stretch, see the [ROS installation instructions](docs/install_robot.md) in `home_robot_hw`.

You may need a calibrated URDF for our inverse kinematics code to work well; see [calibration notes](docs/calibration.md).

#### Network Setup

Follow the [network setup guide](docs/network.md) to set up your robot to use the network, and make sure that it can communicate between workstation and robot via ROS. On the robot side, start up the controllers with:
```
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

### Workstation Instructions

To set up your workstation, follow these instructions. We will assume that your system supports CUDA 11.8 or better for pytorch; earlier versions should be fine, but may require some changes to the conda environment.

#### 1. Create Your Environment
```
# Create a conda env - use the version in home_robot_hw if you want to run on the robot
mamba env create -n home-robot -f src/home_robot_hw/environment.yml

# Otherwise, use the version in src/home_robot
mamba env create -n home-robot -f src/home_robot/environment.yml

conda activate home-robot
```

This should install pytorch; if you run into trouble, you may need to edit the installation to make sure you have the right CUDA version. See the [pytorch install notes](docs/install_pytorch.md) for more.

#### 2. Install Home Robot Packages
```
conda activate home-robot

# Install the core home_robot package
python -m pip install -e src/home_robot

Skip to step 4 if you do not have a real robot setup or if you only want to use our simulation stack.

# Install home_robot_hw
python -m pip install -e src/home_robot_hw
```

_Testing Real Robot Setup:_ Now you can run a couple commands to test your connection. If the `roscore` and the robot controllers are running properly, you can run `rostopic list` and should see a list of topics - streams of information coming from the robot. You can then run RVIZ to visualize the robot sensor output:

```
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
```

#### 3. Download third-party packages
```
git submodule update --init --recursive assets/hab_stretch src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet
```

#### 4. Hardware Testing

Run the hardware manual test to make sure you can control the robot remotely. Ensure the robot has one meter of free space before running the script.

```
python tests/hw_manual_test.py
```

Follow the on-screen instructions. The robot should move through a set of configurations.

#### 5. Install Detic

Install [detectron2](https://detectron2.readthedocs.io/tutorials/install.html). If you installed our default environment above, you may need to [download CUDA11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive).


Download Detic checkpoint as per the instructions [on the Detic github page](https://github.com/facebookresearch/Detic):

```bash
cd $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/detic/Detic/
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate
```

You should be able to run the Detic demo script as per the Detic instructions to verify your installation was correct:
```bash
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

#### 6. Download pretrained skills
```
mkdir -p data/checkpoints
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd ../../
```

#### 7. Simulation Setup

To set up the simulation stack with Habitat, train DDPPO skills and run evaluations: see the [installation instructions](src/home_robot_sim/README.md) in `home_robot_sim`.

For more details on the OVMM challenge, see the [Habitat OVMM readme](projects/habitat_ovmm/README.md).


#### 8. Run Open Vocabulary Mobile Manipulation on Stretch

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
python projects/real_world_ovmm/eval_episode.py
```

## Code Contribution

We welcome contributions to HomeRobot.

There are two main classes in HomeRobot that you need to be concerned with:
  - *Environments* extend the [abstract Environment class](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/abstract_env.py) and provide *observations* of the world, and a way to *apply actions*.
  - *Agents* extend the [abstract Agent class](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/abstract_agent.py), which takes in an [observation](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/interfaces.py#L95) and produces an [action](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/core/interfaces.py#L50).

Generally, new methods will be implemented as Agents.

### Developing on Hardware

See the robot [hardware development guide](docs/hardware_development.md) for some advice that may make developing code on the Stretch easier.

### Organization

[HomeRobot](https://github.com/facebookresearch/home-robot/) is broken up into three different packages:

| Resource | Description |
| -------- | ----------- |
| [home_robot](src/home_robot) | Core package containing agents and interfaces |
| [home_robot_sim](src/home_robot_sim) | OVMM simulation environment based on [AI Habitat](https://aihabitat.org/) |
| [home_robot_hw](src/home_robot_hw) | ROS package containing hardware interfaces for the Hello Robot Stretch |

The [home_robot](src/home_robot) package contains embodiment-agnostic agent code, such as our [ObjectNav agent](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent.py) (finds objects in scenes) and our [hierarchical OVMM agent](https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/agent/ovmm_agent/ovmm_agent.py). These agents can be extended or modified to implement your own solution.

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
