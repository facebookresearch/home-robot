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


class ContinuousNavigationAction:
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError(
                "continuous navigation action space has 3 dimentions, x y and theta"
            )
        self.xyt = xyt


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_NAVIGATION = 1
    CONTINUOUS_MANIPULATION = 2


class HybridAction(Action):
    """Convenience for supporting multiple action types - provides handling to make sure we have the right class at any particular time"""

    action_type: ActionType
    action: Action

    def __init__(self, action):
        """Make sure that we were passed a useful generic action here. Process it into something useful."""
        if type(action) == HybridAction:
            self.action_type = action.action_type
        if type(action) == DiscreteNavigationAction:
            self.action_type = ActionType.DISCRETE
        elif type(action) == ContinuousNavigationAction:
            self.action_type = ActionType.CONTINUOUS_NAVIGATION
        else:
            raise RuntimeError(f"action type{type(action)} not supported")
        self.action = action

    def is_discrete(self):
        """Let environment know if we need to handle a discrete action"""
        return self.action_type == ActionType.DISCRETE

    def is_navigation(self):
        return self.action_type == ActionType.CONTINUOUS_NAVIGATION

    def is_manipulation(self):
        return self.action_type == ActionType.CONTINUOUS_MANIPULATION

    def get(self):
        """Extract continuous component of the command and return it."""
        if self.action_type == ActionType.DISCRETE:
            return self.action
        elif self.action_type == ActionType.CONTINUOUS_NAVIGATION:
            return self.action.xyt
        else:
            return NotImplementedError("we need to support this action type")


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
    joint: Optional[np.array] = None
    relative_resting_position: Optional[np.array] = None
    is_holding: Optional[np.array] = None
    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None
