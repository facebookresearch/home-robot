# Simulation backend for the Hello Stretch

- WIP
- Contains a minimal, dummy simulation for (debugging & automated testing), to be replaced with Habitat simulation
- Habitat simulation: wrapper around [habitat-sim](https://github.com/facebookresearch/habitat-sim), not yet implemented
- Same interface as home_robot_hw

## Installation

After installing [home_robot](../home_robot):

```sh
# Install simulation deps
mamba install -c conda-forge -c ai-habitat habitat-sim

# Install sim
cd src/home_robot_sim
pip install -e .
```

## Usage

### Launching a minimal kinematic simulation (no camera yet)

```sh
python -m home_robot_sim.nodes.fake_stretch_robot
```