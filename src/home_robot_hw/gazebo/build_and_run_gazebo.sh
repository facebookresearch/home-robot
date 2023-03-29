# TODO: cleanup and test
ENV_NAME=stretch_gazebo

# Create conda env
mamba create -n $ENV_NAME -f gazebo_env.yaml
conda activate $ENV_NAME

# Install
git clone https://github.com/hello-robot/stretch_ros
catkin_make --only-pkgs-with-deps stretch_gazebo
source devel/setup.sh

# Run
roslaunch stretch_gazebo gazebo.launch rviz:=true
