#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=7-0:00
#SBATCH --partition=cortex
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -C volta32gb
#SBATCH -o /private/home/xiaohanzhang/Experiments/Exp-%x.out
#SBATCH -e /private/home/xiaohanzhang/Experiments/Exp-%x.err
source ~/.bashrc
conda activate home-robot-minigpt4
cd /private/home/xiaohanzhang/home-robot
cd projects/vlm_planning
python ovmm_data_generation.py --task_id 4
