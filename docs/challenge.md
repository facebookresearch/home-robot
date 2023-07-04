![](docs/HomeRobot_Logo_Horiz_Color_white_bg.png)
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
    FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

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
     
    *Note:* Please, make sure that you keep your local version of `fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023` image up to date with the image we have hosted on [dockerhub](https://hub.docker.com/r/fairembodied/habitat-challenge/tags). This can be done by pruning all cached images, using:
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


