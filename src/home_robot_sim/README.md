# Home-Robot Simulation

## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Supported tasks](#supported-tasks)

## Environment setup

These setup instructions are meant to be followed after reaching step 7 in the main [README.md](../../README.md) file. If you haven't completed those instructions yet, please refer to the main [README.md](../../README.md) and complete the steps mentioned there before continuing.

### On an Ubuntu machine with GPU:


1. Install `habitat_sim` and other dependencies

```
mamba env update -f src/home_robot_sim/environment.yml
```

2. Fetch submodules
```
git submodule update --init --recursive src/third_party/habitat-lab
```

3. Install `habitat-lab`, `habitat-baselines` and `pytorch3d`.
```
python -m pip install -e src/third_party/habitat-lab/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

4. Install home_robot_sim library
```
# Install home robot sim interfaces
pip install -e src/home_robot_sim
```

## Supported tasks

Please follow the links below for instructions on setting up the data directory, training policies and running evaluations for our supported tasks.
1. [Open Vocab Mobile Manipulation (OVMM) in Habitat](../../projects/habitat_ovmm/README.md)
2. [ObjectNav in Habitat](../../projects/habitat_objectnav/README.md)
3. ImageNav in Habitat [WIP]