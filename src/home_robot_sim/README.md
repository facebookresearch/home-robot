# Simulation backend for the Hello Stretch

- WIP
- Contains a minimal, dummy simulation for (debugging & automated testing), to be replaced with Habitat simulation
- Habitat simulation: wrapper around [habitat-sim](https://github.com/facebookresearch/habitat-sim), not yet implemented
- Same interface as home_robot_hw

## Installation
```sh
# Set up Conda environment
mamba env create -f environment.yml
conda activate home_robot_sim

# Install home_robot
cd src/home_robot
pip install -e .

cd -

# Install sim
cd src/home_robot_sim
pip install -e .
```

## Usage

### Launching a minimal kinematic simulation (no camera yet)

```sh
python -m home_robot_sim.nodes.fake_stretch_robot
```