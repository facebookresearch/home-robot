#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# check command-line arguments
yes=false
for arg in "$@"; do
  case "$arg" in
    --yes|-y)
      yes=true
      shift
      ;;
    *)
      # handle other arguments here, if needed
      ;;
  esac
done


echo ""
echo ""
echo "=============================================="
echo "            DOWNLOADING SIM DATA"
echo "=============================================="
echo "Make sure:"
echo " - HOME_ROBOT_ROOT is set"
echo "Currently:"
echo " - HOME_ROBOT_ROOT=$HOME_ROBOT_ROOT"

if [ "$yes" = false ]; then
	echo ""
	read -p "Does all this look correct? (y/n) " yn
	case $yn in
		y ) echo "starting installation...";;
		n ) echo "exiting...";
			exit;;
		* ) echo invalid response!;
			exit 1;;
	esac
fi

cd $HOME_ROBOT_ROOT

# Create data, do not error if it already exists
mkdir -p data/datasets

cd $HOME_ROBOT_ROOT/data
git clone https://huggingface.co/datasets/hssd/hssd-hab --recursive
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects --recursive

cd $HOME_ROBOT_ROOT/data/datasets
git clone https://huggingface.co/datasets/ai-habitat/OVMM_episodes ovmm --recursive

echo ""
echo "Download the robot model Habitat uses..."
mkdir -p $HOME_ROBOT_ROOT/data/robots/hab_stretch
cd $HOME_ROBOT_ROOT/data/robots/hab_stretch
wget --no-check-certificate http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip -o hab_stretch_v1.0.zip
cd $HOME_ROBOT_ROOT


