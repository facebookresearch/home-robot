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

#### Local Evaluation

1. Clone the challenge repository:
   ```
   foo
   ```
1. Implement your own agent or try one of ours. We provide an agents in `path to agents` . For example, `this agent` takes random actions:
   ```
   foo
   ```
1.  Modify the provided Dockerfile (`path to docker file`) if you need custom modifications. Let’s say your code needs `this package`, this dependency should be pip installed inside a conda environment called `home-robot` that is shipped with our HomeRobot challenge docker, as shown below:
    ```dockerfile
    # TODO: update
    FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch"

    ADD agents/agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container using: `docker build . --file docker/{ObjectNav, ImageNav}_random_baseline.Dockerfile -t {objectnav, imagenav}_submission`.
    
    *Note #1:* you may need `sudo` privileges to run this command.
    
    *Note #2:* Please make sure that you keep your local version of `fairembodied/habitat-challenge:habitat_navigation_2023_base_docker` image up to date with the image we have hosted on [dockerhub](https://hub.docker.com/r/fairembodied/habitat-challenge/tags). This can be done by pruning all cached images, using:
    ```
    docker system prune -a
    ```
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agent.py`

1. Scene Dataset. Place this data in: `recommended path to data`
    **Using Symlinks:**  If you used symlinks (i.e. `ln -s`) to link to an existing download of HM3D, there is an additional step. First, make sure there is only one level of symlink (instead of a symlink to a symlink link to a .... symlink) with
      ```bash
      ln -f -s $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2) \
          habitat-challenge-data/data/scene_datasets/hm3d_v0.2
      ```

    Then modify the docker command in `scripts/test_local_{objectnav, imagenav}.sh` file to mount the linked to location by adding `-v $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2):/habitat-challenge-data/data/scene_datasets/hm3d_v0.2`. The modified docker command would be
     ```bash
     # ObjectNav
    docker run \
          -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
          -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2):/habitat-challenge-data/data/scene_datasets/hm3d_v0.2 \
          --runtime=nvidia \
          -e "AGENT_EVALUATION_TYPE=local" \
          -e "TRACK_CONFIG_FILE=/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml" \
          ${DOCKER_NAME}
    
    # ImageNav
    docker run \
          -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
          -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2):/habitat-challenge-data/data/scene_datasets/hm3d_v0.2 \
          --runtime=nvidia \
          -e "AGENT_EVALUATION_TYPE=local" \
          -e "TRACK_CONFIG_FILE=/configs/benchmark/nav/imagenav/imagenav_hm3d_v3_challenge.yaml" \
          ${DOCKER_NAME}
    ```
     
1. Evaluate your docker container locally:
    ```bash
    # Testing ObjectNav
    ./scripts/test_local_objectnav.sh --docker-name objectnav_submission

    # Testing ImageNav
    ./scripts/test_local_imagenav.sh --docker-name imagenav_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2023-03-01 16:35:02,244 distance_to_goal: 6.446822468439738
    2023-03-01 16:35:02,244 success: 0.0
    2023-03-01 16:35:02,244 spl: 0.0
    2023-03-01 16:35:02,244 soft_spl: 0.0014486297806195665
    2023-03-01 16:35:02,244 num_steps: 1.0
    2023-03-01 16:35:02,244 collisions/count: 0.0
    2023-03-01 16:35:02,244 collisions/is_collision: 0.0
    2023-03-01 16:35:02,244 distance_to_goal_reward: 0.0009365876515706381
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.** 
1. If you want to try out one of the controllers we provide, change the `"--action_space"` in the dockerfile (`docker/{ObjectNav, ImageNav}_random_baseline.Dockerfile`) to use either `waypoint_controller` or `discrete_waypoint_controller`.

#### Online Submission

Follow instructions in the `submit` tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI `>= 1.2.3`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.5"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
# ObjectNav
evalai push objectnav_submission:latest --phase <phase-name>

# ImageNav
evalai push imagenav_submission:latest --phase <phase-name>
```

The challenge consists of the following phases:

1. **Minival phase**: This split is the same as the one used in `./scripts/test_local_{objectnav, imagenav}.sh`. The purpose of this phase/split is sanity checking -- to confirm that our remote evaluation reports the same result as the one you’re seeing locally. Each team is allowed maximum of 100 submissions per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers.
1. **Test Standard phase**: The purpose of this phase/split is to serve as the public leaderboard establishing the state of the art; this is what should be used to report results in papers. Each team is allowed maximum of 10 submissions per day for this phase, but again, please use them judiciously. Don’t overfit to the test set.
1. **Test Challenge phase**: This phase/split will be used to decide challenge winners. Each team is allowed a total of 5 submissions until the end of challenge submission phase. The highest performing of these 5 will be automatically chosen. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/).

Note: Your agent will be evaluated on 1000 episodes and will have a total available time of 48 hours to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them by opening an issue on this repository.

### ObjectNav/ImageNav Baselines and DD-PPO Training Starter Code
We have added a config in `configs/ddppo_objectnav_v2_hm3d_stretch.yaml | configs/ddppo_imagenav_v3_hm3d_stretch.yaml` that includes a baseline using DD-PPO from Habitat-Lab.

1. Install the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/) packages. You can install Habitat-Sim using our custom Conda package for habitat challenge 2023 with: ```conda install -c aihabitat habitat-sim-challenge-2023```. For Habitat-Lab, we have created the `habitat-challenge-2023` tag in our Github repo, which can be cloned using: ```git clone --branch challenge-2023 https://github.com/facebookresearch/habitat-lab.git```. Please ensure that both habitat-lab and habitat-baselines packages are installed using ```pip install -e habitat-lab``` and ```pip install -e habitat-baselines```. You will find further information for installation in the Github repositories. 

1. Download the HM3D scene dataset following the instructions [here](https://matterport.com/partners/facebook). After downloading extract the dataset to folder `habitat-lab/data/scene_datasets/hm3d_v0.2/` folder (this folder should contain the `.glb` files from HM3D). Note that the `habitat-lab` folder is the [habitat-lab](https://github.com/facebookresearch/habitat-lab/) repository folder. You could also just symlink to the path of the HM3D scenes downloaded in step-4 of local-evaluation under the `habitat-challenge/habitat-challenge-data/data/scene_datasets` folder. This can be done using `ln -s /path/to/habitat-challenge-data/data/scene_datasets /path/to/habitat-lab/data/scene_datasets/` (if on OSX or Linux).

1. **ObjectNav**: Download the episodes dataset for HM3D ObjectNav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip) and place it in the folder `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d`. If placed correctly, you should have the train and val splits at `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d/v2/train/` and `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d/v2/val/` respectively.

    **ImageNav** Download the episodes dataset for HM3D InstanceImageNav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip) and place it in the folder `habitat-challenge/habitat-challenge-data/data/datasets/instance_imagenav/hm3d`. If placed correctly, you should have the train and val splits at `habitat-challenge/habitat-challenge-data/data/datasets/instance_imagenav/hm3d/v3/train/` and `habitat-challenge/habitat-challenge-data/data/datasets/instance_imagenav/hm3d/v3/val/` respectively.

1. An example on how to train DD-PPO model can be found in [habitat-lab/habitat-baselines/habitat_baselines/rl/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/rl/ddppo). See the corresponding README in habitat-lab for how to adjust the various hyperparameters, save locations, visual encoders and other features.

    1. To run on a single machine use the script [single_node.sh](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/single_node.sh) from the `habitat-lab` directory, where `$task={objectnav_v2, imagenav_v3}`:
        ```bash
        #/bin/bash

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        set -x

        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node 1 \
            habitat_baselines/run.py \
            --config-name=configs/ddppo_${task}_hm3d_stretch.yaml
        ```
    1. There is also an example script named [multi_node_slurm.sh](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/multi_node_slurm.sh) for running the code in distributed mode on a cluster with SLURM. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change ```#SBATCH --nodes $NUM_OF_MACHINES``` to the number of machines and ```#SBATCH --ntasks-per-node $NUM_OF_GPUS``` and ```$SBATCH --gpus $NUM_OF_GPUS``` to specify the number of GPUS to use per requested machine.
        ```bash
        #!/bin/bash
        #SBATCH --job-name=ddppo
        #SBATCH --output=logs.ddppo.out
        #SBATCH --error=logs.ddppo.err
        #SBATCH --gpus 1
        #SBATCH --nodes 1
        #SBATCH --cpus-per-task 10
        #SBATCH --ntasks-per-node 1
        #SBATCH --mem=60GB
        #SBATCH --time=72:00:00
        #SBATCH --signal=USR1@90
        #SBATCH --requeue
        #SBATCH --partition=dev

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
        export MAIN_ADDR

        set -x
        srun python -u -m habitat_baselines.run \
            --config-name=configs/ddppo_${task}_hm3d_stretch.yaml
        ```

1. The checkpoint specified by ```$PATH_TO_CHECKPOINT ``` can evaluated based on the SPL and other measurements by running the following command:

    ```bash
    python -u -m habitat_baselines.run \
        --config-name=configs/ddppo_${task}_hm3d_stretch.yaml \
        habitat_baselines.evaluate=True \
        habitat_baselines.eval_ckpt_path_dir=$PATH_TO_CHECKPOINT \
        habitat.dataset.data_path.split=val
    ```
    The weights used for our DD-PPO Objectnav or Imagenav baseline for the Habitat-2023 challenge can be downloaded with the following command:
    ```bash
    wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/{task}_baseline_habitat_navigation_challenge_2023.pth
    ```
    where `$task={objectnav, imagenav}`.

1. To submit your entry via EvalAI, you will need to build a docker file. We provide Dockerfiles ready to use with the DD-PPO baselines in `docker/{ObjectNav, ImageNav}_ddppo_baseline.Dockerfile`. For the sake of completeness, we describe how you can make your own Dockerfile below. If you just want to test the baseline code, feel free to skip this bullet because  ```ObjectNav_ddppo_baseline.Dockerfile``` is ready to use.
    1. You may want to modify the `{ObjectNav, ImageNav}_ddppo_baseline.Dockerfile` to include PyTorch or other libraries. To install pytorch, ifcfg and tensorboard, add the following command to the Docker file:
        ```dockerfile
        RUN /bin/bash -c ". activate habitat; pip install ifcfg torch tensorboard"
        ```
    1. You change which ```agent.py``` and which ``submission.sh`` script is used in the Docker, modify the following lines and replace the first agent.py or submission.sh with your new files:
        ```dockerfile
        ADD agents/agent.py agent.py
        ADD submission.sh submission.sh
        ```
    1. Do not forget to add any other files you may need in the Docker, for example, we add the ```demo.ckpt.pth``` file which is the saved weights from the DD-PPO example code.

    1. Finally, modify the submission.sh script to run the appropriate command to test your agents. The scaffold for this code can be found in ```agent.py``` and the DD-PPO specific agent can be found in ```habitat_baselines_agents.py```. In this example, we only modify the final command of the ObjectNav/ImageNav docker: by adding the following args to submission.sh ```--model-path demo.ckpt.pth --input-type rgbd```. The default submission.sh script will pass these args to the python script. You may also replace the submission.sh.

1. Once your Dockerfile and other code is modified to your satisfaction, build it with the following command.
    ```bash
    docker build . --file docker/{ObjectNav, ImageNav}_ddppo_baseline.Dockerfile -t {objectnav, imagenav}_submission
    ```
1. To test locally simple run the ```scripts/test_local_{objectnav, imagenav}.sh``` script. If the docker runs your code without errors, it should work on Eval-AI. The instructions for submitting the Docker to EvalAI are listed above.
1. Happy hacking!

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
