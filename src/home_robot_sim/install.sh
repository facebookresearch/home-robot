#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo ""
echo ""
echo "=============================================="
echo "           INSTALLING SIMULATION"
echo "=============================================="
echo "Make sure:"
echo " - Conda env is active"
echo " - CUDA_HOME is set"
echo " - HOME_ROBOT_ROOT is set"
echo "Currently:"
echo " - CUDA_HOME=$CUDA_HOME"
echo " - HOME_ROBOT_ROOT=$HOME_ROBOT_ROOT"
echo " - python=`which python`"
echo ""
read -p "Does all this look correct? (y/n) " yn
case $yn in
	y ) echo "Starting installation...";;
	n ) echo "Exiting...";
		exit;;
	* ) echo Invalid response!;
		exit 1;;
esac

echo "We currently recommend installing simulation from source."
conda env update -f $HOME_ROBOT_ROOT/src/home_robot_sim/environment.yml

cd $HOME_ROBOT_ROOT
pip install -e $HOME_ROBOT_ROOT/src/home_robot_sim

echo ""
echo "Install habitat dependencies..."
cd $HOME_ROBOT_ROOT
git submodule update --init --recursive src/third_party/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines

echo ""
echo "Download data from submodules..."
cd $HOME_ROBOT_ROOT
git submodule update --init --recursive data/objects
