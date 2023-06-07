import cv2
import numpy as np


class GpsVizualizer:
    """
    Visualize the GPS on a map
    """

    def __init__(self, resolution, scaling_factor):
        self._resolution = resolution
        self._scaling_factor = scaling_factor
        self.reset()

    def reset(self):
        self._map = np.full(
            (self._resolution, self._resolution, 3), 255, dtype=np.uint8
        )
        self._poses = [(0, 0)]
        self._gt_poses = [(0, 0)]

    def _get_cur_gt_pose(self, prev_gt_pose, action):
        """
        Get the current ground truth pose from the previous one and the action
        Action is [dx, dy, dtheta] relative to the previous pose
        Returns [x, y, theta] in the global frame
        """
        dx, dy, dtheta = action
        x, y, theta = prev_gt_pose

        # rotate the action vector to the global frame
        dx_g = dx * np.cos(theta) - dy * np.sin(theta)
        dy_g = dx * np.sin(theta) + dy * np.cos(theta)

        # add the action vector to the previous pose
        x_g = x + dx_g
        y_g = y + dy_g
        theta_g = theta + dtheta

        return [x_g, y_g, theta_g]

    def _render_point(self, point, color):
        """
        Render a point on the map
        """
        x = int(point[0] * self._scaling_factor + self._resolution / 2)
        y = int(-point[1] * self._scaling_factor + self._resolution / 2)
        cv2.circle(self._map, (x, y), 3, color, -1)

    def visualize(self, gps, action):
        """
        Visualize the gps on the map
        Origin is at the center of the map
        """
        # Current in red, previous in black
        self._render_point(self._poses[-1], (0, 0, 0))
        self._render_point(gps, (0, 0, 255))
        self._poses.append(gps)

        # Ground truth in green, previous in yellow
        gt_pose = self._get_cur_gt_pose(self._gt_poses[-1], action)
        self._render_point(self._gt_poses[-1][:2], (0, 255, 255))
        self._gt_poses.append(gt_pose)
        self._render_point(gt_pose[:2], (0, 255, 0))

        cv2.imshow("GPS drift", self._map)
        cv2.waitKey(1)
