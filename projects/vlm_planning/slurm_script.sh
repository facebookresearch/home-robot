#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=learnfair
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -C volta32gb
#SBATCH -o /private/home/xiaohanzhang/Experiments/Exp-%x.out
#SBATCH -e /private/home/xiaohanzhang/Experiments/Exp-%x.err
source ~/.bashrc
conda activate home-robot-minigpt4
cd /private/home/xiaohanzhang/home-robot
python projects/vlm_planning/eval_baselines_agent.py

