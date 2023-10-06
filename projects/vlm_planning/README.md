# vlm_planning

## Setup
```
mamba env create -n home-robot-minigpt4 -f projects/vlm_planning/environment.yml
conda activate home-robot-minigpt4
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/minigpt4/cortex-minigpt4
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/minigpt4/Llama-2-7b-chat-hf
```

## Usage
```
python projects/vlm_planning/eval_baselines_agent.py
```
