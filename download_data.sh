#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo ""
echo ""
echo "=============================================="
echo "            DOWNLOADING SIM DATA"
echo "=============================================="
echo "make sure:"
echo " - home_robot_root is set"
echo "currently:"
echo " - home_robot_root=$home_robot_root"
echo ""
read -p "does all this look correct? (y/n) " yn
case $yn in
	y ) echo "starting installation...";;
	n ) echo "exiting...";
		exit;;
	* ) echo invalid response!;
		exit 1;;
esac

cd $HOME_ROBOT_ROOT

# Create data, do not error if it already exists
mkdir -p data/datasets

cd $HOME_ROBOT_ROOT/data
git clone https://huggingface.co/datasets/hssd/hssd-hab --recursive
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects --recursive

cd $HOME_ROBOT_ROOT/data/datasets
git clone https://huggingface.co/datasets/ai-habitat/OVMM_episodes ovmm --recursive

