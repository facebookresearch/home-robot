import numpy as np

from home_robot.motion.stretch import HelloStretchKinematics

# Create bullet client
from home_robot.utils.bullet import PbArticulatedObject, PbClient, PbObject
from home_robot.utils.point_cloud import show_point_cloud

data = np.load("test.npz")
all_xyz = data["all_xyz"]
xyz = data["xyz"]
rgb = data["rgb"]
pt = data["pt"]

show_point_cloud(all_xyz, rgb / 255, orig=np.zeros(3))

# TODO: rotate it 180 degrees

# show_point_cloud(xyz, rgb / 255, orig=pt)

client = PbClient(visualize=True)
red_block = PbObject(
    "red_block", "./assets/red_block.urdf", start_pos=pt, client=client.id
)
PLANNER_STRETCH_URDF = "assets/hab_stretch/urdf/planner_calibrated.urdf"
MANIP_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_manip_mode.urdf"
# Load a robot model here
robot = client.add_articulated_object("robot", PLANNER_STRETCH_URDF)

# Inverse kinematics

model = HelloStretchKinematics()

input("press enter when done")
