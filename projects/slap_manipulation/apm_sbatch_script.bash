#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4320
#SBATCH --partition=learnfair
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=30gb
#SBATCH --constraint=pascal
#SBATCH -o /private/home/wenxuanz/Experiments/slurm_output/Exp%x-%a_%A.out
#SBATCH -e /private/home/wenxuanz/Experiments/slurm_output/Exp%x-%a_%A.err
#SBATCH --job-name=3034
#SBATCH --array=0-2

exec zsh
echo $HOSTNAME
conda activate slap-home-robot
export LD_LIBRARY_PATH=/public/lib::/public/slurm/22.05.6/lib:/private/home/priparashar/mambaforge/envs/slap-home-robot/lib
cd /private/home/priparashar/development/slap_stretch/home-robot/projects/slap_manipulation/

case $SLURM_ARRAY_TASK_ID in
	0)
		python src/slap_manipulation/policy/action_prediction_module.py weights.position=0.1 weights.orientation=0.1 weights.gripper=1e-3
		;;
	1)
		python src/slap_manipulation/policy/action_prediction_module.py weights.position=1 weights.orientation=0.01 weights.gripper=1e-4
		;;
	2)
		python src/slap_manipulation/policy/action_prediction_module.py weights.position=1 weights.orientation=0.01 weights.gripper=1e-2
		;;
esac
