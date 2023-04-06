from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class Action:
    """Controls."""

    pass


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    EXTEND_ARM = 8
    EMPTY_ACTION = 9
    SNAP_OBJECT = 10
    DESNAP_OBJECT = 11
    FACE_ARM = 12
    RESET_JOINTS = 13


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class Observations:
    """Sensor observations."""

    # --------------------------------------------------------
    # Typed observations
    # --------------------------------------------------------

    # Joint states
    # joint_positions: np.ndarray

    # Pose
    # TODO: add these instead of gps + compass
    # base_pose: Pose
    # ee_pose: Pose

    # Pose
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left
    compass: np.ndarray  # positive theta is rotation to left - consistent with robot

    # Camera
    rgb: np.ndarray  # (camera_height, camera_width, 3) in [0, 255]
    depth: np.ndarray  # (camera_height, camera_width) in meters
    xyz: Optional[
        np.ndarray
    ] = None  # (camera_height, camera_width, 3) in camera coordinates
    semantic: Optional[
        np.array
    ] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]
    third_person_image: Optional[np.array] = None
    camera_pose: Optional[np.array] = None
    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None
