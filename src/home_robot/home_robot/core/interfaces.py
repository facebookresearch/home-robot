from typing import Optional, Any, Dict
import numpy as np
from enum import Enum
from dataclasses import dataclass


class Action:
    """Controls."""
    pass


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


@dataclass
class Observations:
    """Sensor observations."""

    # --------------------------------------------------------
    # Typed observations
    # --------------------------------------------------------

    # Pose
    compass: np.array
    gps: np.array

    # Camera
    rgb: np.ndarray  # (camera_height, camera_width, 3) in [0, 255]
    depth: np.ndarray  # (camera_height, camera_width) in meters
    semantic: Optional[np.array] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]

    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None
