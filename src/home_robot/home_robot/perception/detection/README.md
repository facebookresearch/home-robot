# Detectors

## Table of contents
   1. [DETIC](#detic)
   2. [MaskRCNN](#maskrcnn)
   3. [Grounded-MobileSAM](#grounded_sam)


We support multiple state-of-the-art detectors for robotic perception. Follow the instructions below for setting them up. To switch between the detectors set `detection_module` in agent configs.

## DETIC
```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic
```

Follow the instructions for installing DETIC on home-robot [README](../../../README.md)

## MaskRCNN
TODO


## Mobile GroundedSAM

### Download submodules
```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/grounded_sam/Grounded-Segment-Anything
```

### Install dependencies
Follow the instructions [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/EfficientSAM#installation) for installing `Grounded-MobileSAM`

### Download checkpoints
Place the following under $HOME_ROBOT_ROOT folder
- Download the MobileSAM weights from [here](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights)
- Download the GroundingDINO weights using `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`