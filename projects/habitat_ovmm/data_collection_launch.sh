#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --constraint volta32gb
#SBATCH -o /private/home/xiaohanzhang/Experiments/Exp-%x.out
#SBATCH -e /private/home/xiaohanzhang/Experiments/Exp-%x.err
source ~/.bashrc
conda activate home-robot
cd $HOME_ROBOT_ROOT
python projects/habitat_ovmm/eval_baselines_agent.py --agent_type explore --force_step 600 --data_dir="/private/home/xiaohanzhang/data/eplan" habitat.dataset.episode_indices_range=[$1,$(($1 + 50))]