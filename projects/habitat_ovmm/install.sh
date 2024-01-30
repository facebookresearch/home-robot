#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Download the scenes
cd $HOME_ROBOT_ROOT
git submodule update --init --recursive

echo "Download the scenes..."
git submodule update --init data/hssd-hab

# Download the objects and metadata
cd $HOME_ROBOT_ROOT
echo "Download the objects..."
git submodule update --init data/objects

cd $HOME_ROBOT_ROOT
echo "Download the episodes..."
git submodule update --init data/datasets/ovmm

echo "Call the data download script!"
echo "Safety; make sure you have downloaded everything."
echo "If these are empty, you may have too old a git version."
cd $HOME_ROBOT_ROOT
./download_data.sh

echo "Download the robot model..."
mkdir -p $HOME_ROBOT_ROOT/data/robots/hab_stretch
cd $HOME_ROBOT_ROOT/data/robots/hab_stretch
echo $HOME_ROBOT_ROOT/data/robots/hab_stretch

echo "Unzip the robot model..."
wget http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip hab_stretch_v1.0.zip

echo "Done!"
