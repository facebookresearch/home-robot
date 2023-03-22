# Home Robot Core Package

## Installation

For installing on server-side:
```sh
cd $HOME_ROBOT_ROOT/src/home_robot
mamba env create -n home_robot -f environment.yml
conda activate home_robot
pip install -e .
```

For installing on robot-side:
```sh
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -e .
```