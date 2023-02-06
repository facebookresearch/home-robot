from typing import Optional, Any, Dict
import numpy as np
from enum import Enum


class Action:
    """Controls."""
    pass


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Observations:
    """Sensor observations."""

    def __init__(self,
                 rgb: np.array,
                 depth: np.array,
                 semantic: Optional[np.array],
                 compass: np.array,
                 gps: np.array,
                 task_observations: Dict[str, Any]):
        """
        Arguments:
            rgb: (camera_height, camera_width, 3) in [0, 255]
            depth: (camera_height, camera_width) in meters
            semantic: (camera_height, camera_width) in [0, num_sem_categories - 1]
            compass: base yaw in radians in [-pi, pi] (relative to episode start)
            gps: base (x, y) position in meters (relative to episode start)
            task_observations: untyped task_specific observations
        """
        self.rgb = rgb
        self.depth = depth
        self.semantic = semantic
        self.compass = compass
        self.gps = gps
        self.task_observations = task_observations
