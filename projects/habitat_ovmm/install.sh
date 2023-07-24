#!/usr/bin/env bash

# Download the scenes
cd $HOME_ROBOT_ROOT
git submodule update --init data/hssd-hab

# Download the objects and metadata
cd $HOME_ROBOT_ROOT
git submodule update --init data/objects

cd $HOME_ROBOT_ROOT
git submodule update --init data/datasets/ovmm

cd $HOME_ROBOT_ROOT/data/hssd-hab
git lfs pull
cd -

cd $HOME_ROBOT_ROOT/data/objects
git lfs pull
cd -

cd $HOME_ROBOT_ROOT/data/datasets/ovmm
git lfs pull
cd -

mkdir -p $HOME_ROBOT_ROOT/data/robots/hab_stretch
cd $HOME_ROBOT_ROOT/data/robots/hab_stretch

wget http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip hab_stretch_v1.0.zip
