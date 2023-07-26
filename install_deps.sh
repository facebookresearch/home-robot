#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo ""
echo ""
echo "=============================================="
echo "           INSTALLING DEPENDENCIES"
echo "=============================================="
echo "Make sure:"
echo " - Conda env is active"
echo " - CUDA_HOME is set"
echo " - HOME_ROBOT_ROOT is set"
echo "Currently:"
echo " - CUDA_HOME=$CUDA_HOME"
echo " - HOME_ROBOT_ROOT=$HOME_ROBOT_ROOT"
echo " - python=`which python`"

# Activate conda environment
# conda activate $ENV

echo ""
echo "Install home_robot core..."
python -m pip install -e src/home_robot
echo "Install home_robot ROS..."
python -m pip install -e src/home_robot_hw

echo ""
echo "Install habitat dependencies..."
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines

echo ""
echo "Install detectron2..."
git submodule update --init --recursive src/third_party/detectron2
cd $HOME_ROBOT_ROOT
pip install -e src/third_party/detectron2

echo ""
echo "Downloading pretrained skills..."
cd $HOME_ROBOT_ROOT
mkdir -p $HOME_ROBOT_ROOT/data/checkpoints
cd $HOME_ROBOT_ROOT/data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd $HOME_ROBOT_ROOT
