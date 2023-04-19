import numpy as np

from home_robot.utils.point_cloud import show_point_cloud


class PlacePlanner:
    def __init__(self, robot_client):
        self.robot_client = robot_client

    def try_placing(self, obj_xyz, obj_rgb):
        show_point_cloud(obj_xyz, obj_rgb / 255.0, orig=np.zeros(3))
