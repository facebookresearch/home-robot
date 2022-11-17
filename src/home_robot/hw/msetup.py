import mrp

from home_robot.utils.mrp.envs import catkin_ws_cmd

# Launches these nodes:
# - stretch_core/launch/stretch_driver.launch
# - stretch_core/launch/rplidar.launch
# - stretch_core/launch/stretch_scan_matcher.launch
mrp.process(
    name="stretch_core",
    runtime=mrp.Host(
        run_command=[catkin_ws_cmd, "&&", "roslaunch", "home_robot", "stretch_laser_odom_base.launch"],
    ),
)

# Hector SLAM
mrp.process(
    name="stretch_hector_slam",
    runtime=mrp.Host(
        run_command=[catkin_ws_cmd, "&&", "roslaunch", "home_robot", "stretch_hector_slam.launch"],
    ),
)

if __name__ == "__main__":
    mrp.main()
