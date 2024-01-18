mamba env create -n home-robot -f src/home_robot_spot/env.yaml
conda activate home-robot
mamba env update -f src/home_robot_hw/environment.yml
./install_deps.sh
