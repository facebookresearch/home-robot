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
echo ""
read -p "Does all this look correct? (y/n) " yn
case $yn in
	y ) echo "Starting installation...";;
	n ) echo "Exiting...";
		exit;;
	* ) echo Invalid response!;
		exit 1;;
esac

# Activate conda environment
# conda activate $ENV

echo ""
echo "Ensure Git LFS is installed"
git lfs install

echo ""
echo "Install home_robot core..."
python -m pip install -e src/home_robot
echo "Install home_robot ROS..."
python -m pip install -e src/home_robot_hw

echo "WARNING! we should include this elsewhere"
python -m pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+118.html

echo ""
echo "Install habitat dependencies..."
git submodule update --init --recursive src/third_party/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines

echo ""
echo "Install detectron2..."
git submodule update --init --recursive src/third_party/detectron2
cd $HOME_ROBOT_ROOT
pip install -e src/third_party/detectron2

cd $HOME_ROBOT_ROOT
echo ""
echo "Cloning Detic and third party submodules..."
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic

echo ""
echo "Download DETIC checkpoint..."
cd $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/detic/Detic
mkdir models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

cd $HOME_ROBOT_ROOT
echo ""
echo "Cloning GroundedSAM third party submodule..."
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/grounded_sam/Grounded-Segment-Anything

echo "Installing GroundedSAM dependencies..."
cd $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
cd grounded-sam-osx && bash install.sh
python -m pip install opencv-python pycocotools matplotlib ipykernel

mkdir $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/checkpoints/

echo "Downloading MobileSAM checkpoint..."
wget --no-check-certificate -P $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/checkpoints/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

echo "Downloading SAM checkpoint..."
wget --no-check-certificate -P $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Downloading GroundedDINO checkpoints.."
wget --no-check-certificate -P $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget --no-check-certificate -P $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/grounded_sam/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth

echo ""
echo "Downloading pretrained skills..."
cd $HOME_ROBOT_ROOT
mkdir -p $HOME_ROBOT_ROOT/data/checkpoints
cd $HOME_ROBOT_ROOT/data/checkpoints
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd $HOME_ROBOT_ROOT

echo ""
echo "Install pre-commit hooks"
pip install pre-commit
pre-commit install
pre-commit install-hooks
