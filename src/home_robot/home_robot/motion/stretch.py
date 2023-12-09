# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
from typing import List, Optional, Tuple

import home_robot.utils.bullet as hrb
import numpy as np
import pybullet as pb
from home_robot.core.interfaces import ContinuousFullBodyAction
from home_robot.motion.bullet import BulletRobotModel, PybulletIKSolver
from home_robot.motion.pinocchio_ik_solver import PinocchioIKSolver, PositionIKOptimizer
from home_robot.motion.robot import Footprint
from home_robot.utils.pose import to_matrix

# Stretch stuff
DEFAULT_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_dex_wrist_simplified.urdf"
PLANNER_STRETCH_URDF = "assets/hab_stretch/urdf/planner_calibrated.urdf"
MANIP_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_manip_mode.urdf"

STRETCH_HOME_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.2,  # lift
        0.057,  # arm
        0.0,  # gripper rpy
        0.0,
        0.0,
        3.0,  # wrist,
        0.0,
        0.0,
    ]
)

# look down in navigation mode for doing manipulation post-navigation
STRETCH_POSTNAV_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-45),
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_DEMO_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper straight out, lowered arm for clear vision
STRETCH_PREDEMO_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        0.0,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)
# Navigation should not be fully folded up against the arm - in case its holding something
STRETCH_NAVIGATION_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-30),
    ]
)


PIN_CONTROLLED_JOINTS = [
    "base_x_joint",
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]


# used for mapping joint states in STRETCH_*_Q to match the sim/real joint action space
def map_joint_q_state_to_action_space(q):
    return np.array(
        [
            q[4],  # arm_0
            q[3],  # lift
            q[8],  # yaw
            q[7],  # pitch
            q[6],  # roll
            q[9],  # head pan
            q[10],  # head tilt
        ]
    )


# This is the gripper, and the distance in the gripper frame to where the fingers will roughly meet
STRETCH_GRASP_FRAME = "link_straight_gripper"
STRETCH_CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"

# Offsets required for "link_straight_gripper" grasp frame
STRETCH_STANDOFF_DISTANCE = 0.235
STRETCH_STANDOFF_WITH_MARGIN = 0.25
# Offset from a predicted grasp point to STRETCH_GRASP_FRAME
STRETCH_GRASP_OFFSET = np.eye(4)
STRETCH_GRASP_OFFSET[:3, 3] = np.array([0, 0, -1 * STRETCH_STANDOFF_DISTANCE])
# Offset from STRETCH_GRASP_FRAME to predicted grasp point
STRETCH_TO_GRASP = np.eye(4)
STRETCH_TO_GRASP[:3, 3] = np.array([0, 0, STRETCH_STANDOFF_DISTANCE])

# Other stretch parameters
STRETCH_GRIPPER_OPEN = 0.22
STRETCH_GRIPPER_CLOSE = -0.2
STRETCH_HEAD_CAMERA_ROTATIONS = (
    3  # number of counterclockwise rotations for the head camera
)

# For EXTEND_ARM action
STRETCH_ARM_EXTENSION = 0.8
STRETCH_ARM_LIFT = 0.8


# Stores joint indices for the Stretch configuration space
class HelloStretchIdx:
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    LIFT = 3
    ARM = 4
    GRIPPER = 5
    WRIST_ROLL = 6
    WRIST_PITCH = 7
    WRIST_YAW = 8
    HEAD_PAN = 9
    HEAD_TILT = 10


class HelloStretchKinematics(BulletRobotModel):
    """Define motion planning structure for the robot. Exposes kinematics."""

    # DEFAULT_BASE_HEIGHT = 0.09
    DEFAULT_BASE_HEIGHT = 0
    GRIPPER_OPEN = 0.6
    GRIPPER_CLOSED = -0.3

    default_step = np.array(
        [
            0.1,
            0.1,
            0.2,  # x y theta
            0.025,
            0.025,  # lift and arm
            0.3,  # gripper
            0.1,
            0.1,
            0.1,  # wrist rpy
            0.2,
            0.2,  # head
        ]
    )
    default_tols = np.array(
        [
            0.1,
            0.1,
            0.01,  # x y theta
            0.001,
            0.0025,  # lift and arm
            0.01,  # gripper
            0.01,
            0.01,
            0.01,  # wrist rpy
            10.0,
            10.0,  # head - TODO handle this better
        ]
    )
    # look_at_ee = np.array([-np.pi/2, -np.pi/8])
    look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    look_front = np.array([0.0, math.radians(-30)])
    look_ahead = np.array([0.0, 0.0])
    look_close = np.array([0.0, math.radians(-45)])

    max_arm_height = 1.2

    # For inverse kinematics mode
    default_ee_link_name = "link_straight_gripper"

    default_manip_mode_controlled_joints = [
        "base_x_joint",
        "joint_lift",
        "joint_arm_l3",
        "joint_arm_l2",
        "joint_arm_l1",
        "joint_arm_l0",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    manip_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]
    full_body_controlled_joints = [
        "base_x_joint",
        "base_y_joint",
        "base_theta_joint",
        "joint_lift",
        "joint_arm_l3",
        "joint_arm_l2",
        "joint_arm_l1",
        "joint_arm_l0",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]

    def get_footprint(self) -> Footprint:
        """Return footprint for the robot. This is expected to be a mask."""
        # Note: close to the actual measurements
        # return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=0.1)
        return Footprint(width=0.4, length=0.5, width_offset=0.0, length_offset=0.1)
        # return Footprint(width=0.2, length=0.2, width_offset=0.0, length_offset=0.1)

    def _create_ik_solvers(self, ik_type: str = "pinocchio", visualize: bool = False):
        """Create ik solvers using physics backends such as pybullet or pinocchio."""
        # You can set one of the visualize flags to true to debug IK issues
        # This is not exposed manually - only one though or it will fail
        assert ik_type in [
            "pybullet",
            "pinocchio",
            "pybullet_optimize",
            "pinocchio_optimize",
        ], f"Unknown ik type: {ik_type}"

        # You can set one of the visualize flags to true to debug IK issues
        self._manip_dof = len(self._manip_mode_controlled_joints)
        if "pybullet" in ik_type:
            ranges = np.zeros((self._manip_dof, 2))
            ranges[:, 0] = self._to_manip_format(self.range[:, 0])
            ranges[:, 1] = self._to_manip_format(self.range[:, 1])
            self.manip_ik_solver = PybulletIKSolver(
                self.manip_mode_urdf_path,
                self._ee_link_name,
                self._manip_mode_controlled_joints,
                visualize=visualize,
                joint_range=ranges,
            )
        elif "pinocchio" in ik_type:
            self.manip_ik_solver = PinocchioIKSolver(
                self.manip_mode_urdf_path,
                self._ee_link_name,
                self._manip_mode_controlled_joints,
            )

        if "optimize" in ik_type:
            self.manip_ik_solver = PositionIKOptimizer(
                ik_solver=self.manip_ik_solver,
                pos_error_tol=0.005,
                ori_error_range=np.array([0.0, 0.0, 0.2]),
            )

    def __init__(
        self,
        name: str = "hello_robot_stretch",
        urdf_path: str = "",
        visualize: bool = False,
        root: str = ".",
        ik_type: str = "pinocchio",
        ee_link_name: Optional[str] = None,
        grasp_frame: Optional[str] = None,
        joint_tolerance: float = 0.01,
        manip_mode_controlled_joints: Optional[List[str]] = None,
    ):
        """Create the robot in bullet for things like kinematics; extract information"""

        self.joint_tol = joint_tolerance

        # urdf
        if not urdf_path:
            full_body_urdf = PLANNER_STRETCH_URDF
            manip_urdf = MANIP_STRETCH_URDF
        else:
            full_body_urdf = os.path.join(urdf_path, "planner_calibrated.urdf")
            manip_urdf = os.path.join(
                urdf_path, "planner_calibrated_manipulation_mode.urdf"
            )
        self.full_body_urdf_path = os.path.join(root, full_body_urdf)
        self.manip_mode_urdf_path = os.path.join(root, manip_urdf)
        super(HelloStretchKinematics, self).__init__(
            name=name, urdf_path=self.full_body_urdf_path, visualize=visualize
        )

        # DOF: 3 for ee roll/pitch/yaw
        #      1 for gripper
        #      1 for ee extension
        #      3 for base x/y/theta
        #      2 for head
        self.dof = 3 + 2 + 4 + 2
        self.joints_dof = 10  # from habitat spec
        self.base_height = self.DEFAULT_BASE_HEIGHT

        # ranges for joints
        self.range = np.zeros((self.dof, 2))

        # Create object reference
        self.set_pose = self.ref.set_pose
        self.set_joint_position = self.ref.set_joint_position

        self._update_joints()

        self._ik_type = ik_type
        self._ee_link_name = (
            ee_link_name if ee_link_name is not None else self.default_ee_link_name
        )
        self._grasp_frame = (
            grasp_frame if grasp_frame is not None else STRETCH_GRASP_FRAME
        )
        self._manip_mode_controlled_joints = (
            manip_mode_controlled_joints
            if manip_mode_controlled_joints is not None
            else self.default_manip_mode_controlled_joints
        )

        self._create_ik_solvers(ik_type=ik_type, visualize=visualize)

    def get_dof(self) -> int:
        """return degrees of freedom of the robot"""
        return self.dof

    def set_head_config(self, q):
        # WARNING: this sets all configs
        bidxs = [HelloStretchIdx.HEAD_PAN, HelloStretchIdx.HEAD_TILT]
        bidxs = [self.joint_idx[b] for b in bidxs]
        if len(q) == self.dof:
            qidxs = bidxs
        elif len(q) == 2:
            qidxs = [0, 1]
        else:
            raise RuntimeError("unsupported number of head joints")

        for idx, qidx in zip(bidxs, qidxs):
            # Idx is the urdf joint index
            # qidx is the idx in the provided query
            qq = q[qidx]
            self.ref.set_joint_position(idx, qq)

    def sample_uniform(self, q0=None, pos=None, radius=2.0):
        """Sample random configurations to seed the ik planner"""
        q = (np.random.random(self.dof) * self._rngs) + self._mins
        q[HelloStretchIdx.BASE_THETA] = np.random.random() * np.pi * 2
        # Set the gripper state
        if q0 is not None:
            q[HelloStretchIdx.GRIPPER] = q0[HelloStretchIdx.GRIPPER]
        # Set the position to sample poses
        if pos is not None:
            x, y = pos[0], pos[1]
        elif q0 is not None:
            x = q0[HelloStretchIdx.BASE_X]
            y = q0[HelloStretchIdx.BASE_Y]
        else:
            x, y = None, None
        # Randomly sample
        if x is not None:
            theta = np.random.random() * 2 * np.pi
            dx = radius * np.cos(theta)
            dy = radius * np.sin(theta)
            q[HelloStretchIdx.BASE_X] = x + dx
            q[HelloStretchIdx.BASE_Y] = y + dy
        return q

    def config_open_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][1]
        return q

    def config_close_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][0]
        return q

    def _update_joints(self):
        """Get joint info from URDF or otherwise provide it"""
        self.joint_idx = [-1] * self.dof
        # Get the joint info we need from this
        joint_lift = self.ref.get_joint_info_by_name("joint_lift")
        self.range[:3, 0] = -float("Inf") * np.ones(3)
        self.range[:3, 1] = float("Inf") * np.ones(3)
        self.range[HelloStretchIdx.LIFT] = np.array(
            [
                joint_lift.lower_limit + self.joint_tol,
                joint_lift.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.LIFT] = joint_lift.index
        joint_head_pan = self.ref.get_joint_info_by_name("joint_head_pan")
        self.range[HelloStretchIdx.HEAD_PAN] = np.array(
            [
                joint_head_pan.lower_limit + self.joint_tol,
                joint_head_pan.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.HEAD_PAN] = joint_head_pan.index
        joint_head_tilt = self.ref.get_joint_info_by_name("joint_head_tilt")
        self.range[HelloStretchIdx.HEAD_TILT] = np.array(
            [joint_head_tilt.lower_limit, joint_head_tilt.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.HEAD_TILT] = joint_head_tilt.index
        joint_wrist_yaw = self.ref.get_joint_info_by_name("joint_wrist_yaw")
        self.range[HelloStretchIdx.WRIST_YAW] = np.array(
            [
                joint_wrist_yaw.lower_limit + self.joint_tol,
                joint_wrist_yaw.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_YAW] = joint_wrist_yaw.index
        joint_wrist_roll = self.ref.get_joint_info_by_name("joint_wrist_roll")
        self.range[HelloStretchIdx.WRIST_ROLL] = np.array(
            [
                joint_wrist_roll.lower_limit + self.joint_tol,
                joint_wrist_roll.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_ROLL] = joint_wrist_roll.index
        joint_wrist_pitch = self.ref.get_joint_info_by_name("joint_wrist_pitch")
        self.range[HelloStretchIdx.WRIST_PITCH] = np.array(
            [
                joint_wrist_pitch.lower_limit + self.joint_tol,
                joint_wrist_pitch.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_PITCH] = joint_wrist_pitch.index

        # arm position
        # TODO: fix this so that it is not hard-coded any more
        self.range[HelloStretchIdx.ARM] = np.array([0.0, 0.75])
        self.arm_idx = []
        upper_limit = 0
        for i in range(4):
            joint = self.ref.get_joint_info_by_name("joint_arm_l%d" % i)
            self.arm_idx.append(joint.index)
            upper_limit += joint.upper_limit

        # TODO: gripper
        self.gripper_idx = []
        for i in ["right", "left"]:
            joint = self.ref.get_joint_info_by_name("joint_gripper_finger_%s" % i)
            self.gripper_idx.append(joint.index)
            print(i, joint.name, joint.lower_limit, joint.upper_limit)
            self.range[HelloStretchIdx.GRIPPER] = (
                np.array([joint.lower_limit, joint.upper_limit]) * 0.5
            )

        self._mins = self.range[:, 0]
        self._maxs = self.range[:, 1]
        self._rngs = self.range[:, 1] - self.range[:, 0]

    def get_backend(self):
        return self.backend

    def get_object(self) -> hrb.PbArticulatedObject:
        """return back-end reference to the Bullet object"""
        return self.ref

    def _set_joint_group(self, idxs, val):
        for idx in idxs:
            self.ref.set_joint_position(idx, val)

    def vanish(self):
        """get rid of the robot"""
        self.ref.set_pose([0, 0, 1000], [0, 0, 0, 1])

    def set_config(self, q):
        assert len(q) == self.dof
        x, y, theta = q[:3]
        # quaternion = pb.getQuaternionFromEuler((0, 0, theta))
        self.ref.set_pose((0, 0, self.base_height), [0, 0, 0, 1])
        # self.ref.set_pose((x, y, self.base_height), quaternion)
        self.ref.set_joint_position(0, x)
        self.ref.set_joint_position(1, y)
        self.ref.set_joint_position(2, theta)
        for idx, qq in zip(self.joint_idx, q):
            if idx < 0:
                continue
            self.ref.set_joint_position(idx, qq)
        # Finally set the arm and gripper as groups
        self._set_joint_group(self.arm_idx, q[HelloStretchIdx.ARM] / 4.0)
        self._set_joint_group(self.gripper_idx, q[HelloStretchIdx.GRIPPER])

    def plan_look_at(self, q0, xyz):
        """assume this is a relative xyz"""
        dx, dy = xyz[:2] - q0[:2]
        theta0 = q0[2]
        thetag = np.arctan2(dy, dx)
        action = np.zeros(self.dof)

        dist = np.abs(thetag - theta0)
        # Dumb check to see if we can get the rotation right
        if dist > np.pi:
            thetag -= np.pi / 2
            dist = np.abs(thetag - theta0)
        # Getting the direction right here
        dirn = 1.0 if thetag > theta0 else -1.0
        action[2] = dist * dirn

        look_action = q0.copy()
        self.update_look_ahead(look_action)

        # Compute the angle
        head_height = 1.2
        distance = np.linalg.norm([dx, dy])

        look_action[HelloStretchIdx.HEAD_TILT] = -1 * np.arctan(
            (head_height - xyz[2]) / distance
        )

        return [action, look_action]

    def interpolate(self, q0, qg, step=None, xy_tol=0.05, theta_tol=0.01):
        """interpolate from initial to final configuration. for this robot we break it up into
        four stages:
        1) rotate to point towards final location
        2) drive to final location
        3) rotate to final orientation
        4) move everything else
        """
        if step is None:
            step = self.default_step
        qi = q0.copy()
        theta0 = q0[HelloStretchIdx.BASE_THETA]
        thetag = qg[HelloStretchIdx.BASE_THETA]
        xy0 = q0[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        xyg = qg[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        dist = np.linalg.norm(xy0 - xyg)
        if dist > xy_tol:
            dx, dy = xyg - xy0
            theta = np.arctan2(dy, dx)
            for qi, ai in self.interpolate_angle(
                qi, theta0, theta, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
            for qi, ai in self.interpolate_xy(
                qi, xy0, dist, step[HelloStretchIdx.BASE_X]
            ):
                yield qi, ai
        else:
            theta = theta0
        # update angle
        if np.abs(thetag - theta) > theta_tol:
            for qi, ai in self.interpolate_angle(
                qi, theta, thetag, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
        # Finally interpolate the whole joint space
        for qi, ai in self.interpolate_arm(qi, qg, step):
            yield qi, ai

    def get_link_pose(self, link_name, q=None):
        if q is not None:
            self.set_config(q)
        return self.ref.get_link_pose(link_name)

    def manip_fk(self, q: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """manipulator specific forward kinematics; uses separate URDF than the full-body fk() method"""
        assert q.shape == (self.dof,)

        if "pinocchio" in self._ik_type:
            q = self._ros_pose_to_pinocchio(q)

        ee_pos, ee_quat = self.manip_ik_solver.compute_fk(q)
        return ee_pos.copy(), ee_quat.copy()

    def fk(self, q=None, as_matrix=False) -> Tuple[np.ndarray, np.ndarray]:
        """forward kinematics"""
        pose = self.get_link_pose(self.ee_link_name, q)
        if as_matrix:
            return to_matrix(*pose)
        return pose

    def update_head(self, qi: np.ndarray, look_at) -> np.ndarray:
        """move head based on look_at and return the joint-state"""
        qi[HelloStretchIdx.HEAD_PAN] = look_at[0]
        qi[HelloStretchIdx.HEAD_TILT] = look_at[1]
        return qi

    def update_gripper(self, qi, open=True):
        """update target state for gripper"""
        if open:
            qi[HelloStretchIdx.GRIPPER] = STRETCH_GRIPPER_OPEN
        else:
            qi[HelloStretchIdx.GRIPPER] = STRETCH_GRIPPER_CLOSE
        return qi

    def interpolate_xy(self, qi, xy0, dist, step=0.1):
        """just move forward with step to target distance"""
        # create a trajectory here
        x, y = xy0
        theta = qi[HelloStretchIdx.BASE_THETA]
        while dist > 0:
            qi = self.update_head(qi.copy(), self.look_front)
            ai = np.zeros(self.dof)
            if dist > step:
                dx = step
            else:
                dx = dist
            dist -= dx
            x += np.cos(theta) * dx
            y += np.sin(theta) * dx
            # x += np.sin(theta) * dx
            # y += np.cos(theta) * dx
            qi[HelloStretchIdx.BASE_X] = x
            qi[HelloStretchIdx.BASE_Y] = y
            ai[0] = dx
            yield qi, ai

    def _to_ik_format(self, q):
        qi = np.zeros(self.ik_solver.get_num_joints())
        qi[0] = q[HelloStretchIdx.BASE_X]
        qi[1] = q[HelloStretchIdx.BASE_Y]
        qi[2] = q[HelloStretchIdx.BASE_THETA]
        qi[3] = q[HelloStretchIdx.LIFT]
        # Next 4 are all arm joints
        arm_ext = q[HelloStretchIdx.ARM] / 4.0
        qi[4] = arm_ext
        qi[5] = arm_ext
        qi[6] = arm_ext
        qi[7] = arm_ext
        # Wrist joints
        qi[8] = q[HelloStretchIdx.WRIST_YAW]
        qi[9] = q[HelloStretchIdx.WRIST_PITCH]
        qi[10] = q[HelloStretchIdx.WRIST_ROLL]
        return qi

    def _to_manip_format(self, q):
        qi = np.zeros(self._manip_dof)
        qi[0] = q[HelloStretchIdx.BASE_X]
        qi[1] = q[HelloStretchIdx.LIFT]
        # Next 4 are all arm joints
        arm_ext = q[HelloStretchIdx.ARM] / 4.0
        qi[2] = arm_ext
        qi[3] = arm_ext
        qi[4] = arm_ext
        qi[5] = arm_ext
        # Wrist joints
        qi[6] = q[HelloStretchIdx.WRIST_YAW]
        qi[7] = q[HelloStretchIdx.WRIST_PITCH]
        qi[8] = q[HelloStretchIdx.WRIST_ROLL]
        return qi

    def _to_plan_format(self, q):
        qi = np.zeros(self.dof)
        qi[HelloStretchIdx.BASE_X] = q[0]
        qi[HelloStretchIdx.BASE_Y] = q[1]
        qi[HelloStretchIdx.BASE_THETA] = q[2]
        qi[HelloStretchIdx.LIFT] = q[3]
        # Arm is sum of the next four joints
        qi[HelloStretchIdx.ARM] = q[4] + q[5] + q[6] + q[7]
        qi[HelloStretchIdx.WRIST_YAW] = q[8]
        qi[HelloStretchIdx.WRIST_PITCH] = q[9]
        qi[HelloStretchIdx.WRIST_ROLL] = q[10]
        return qi

    def _from_manip_format(self, q_raw, q_init):
        # combine arm telescoping joints
        # This is sort of an action representation
        # Compute the actual robot conmfiguration
        q = q_init.copy()
        # Get the theta - we can then convert this over to see where the robot will end up
        q[HelloStretchIdx.BASE_X] = q_raw[0]
        # q[HelloStretchIdx.BASE_Y] += 0
        # No change to theta
        q[HelloStretchIdx.LIFT] = q_raw[1]
        q[HelloStretchIdx.ARM] = np.sum(q_raw[2:6])
        q[HelloStretchIdx.WRIST_ROLL] = q_raw[8]
        q[HelloStretchIdx.WRIST_PITCH] = q_raw[7]
        q[HelloStretchIdx.WRIST_YAW] = q_raw[6]
        return q

    def _pinocchio_pose_to_ros(self, joint_angles):
        raise NotImplementedError

    def _ros_pose_to_pinocchio(self, joint_angles):
        """utility to convert Stretch joint angle output to pinocchio joint pose format"""
        pin_compatible_joints = np.zeros(9)
        pin_compatible_joints[0] = joint_angles[HelloStretchIdx.BASE_X]
        pin_compatible_joints[1] = joint_angles[HelloStretchIdx.LIFT]
        pin_compatible_joints[2] = pin_compatible_joints[3] = pin_compatible_joints[
            4
        ] = pin_compatible_joints[5] = (joint_angles[HelloStretchIdx.ARM] / 4)
        pin_compatible_joints[6] = joint_angles[HelloStretchIdx.WRIST_YAW]
        pin_compatible_joints[7] = joint_angles[HelloStretchIdx.WRIST_PITCH]
        pin_compatible_joints[8] = joint_angles[HelloStretchIdx.WRIST_ROLL]
        return pin_compatible_joints

    def ik(self, pose, q0):
        pos, rot = pose
        se3 = pb.getMatrixFromQuaternion(rot)
        pose = np.eye(4)
        pose[:3, :3] = np.array(se3).reshape(3, 3)
        x, y, z = pos
        pose[:3, 3] = np.array([x, y, z - self.base_height])
        q, success, debug_info = self.ik_solver.compute_ik(pose, self._to_ik_format(q0))
        if q is not None and success:
            return self._to_plan_format(q)
        else:
            return None

    def manip_ik(
        self,
        pose_query,
        q0=None,
        relative: bool = True,
        update_pb: bool = True,
        num_attempts: int = 1,
        verbose: bool = False,
    ):
        """IK in manipulation mode. Takes in a 4x4 pose_query matrix in se(3) and initial
        configuration of the robot.

        By default move relative. easier that way.
        """

        if q0 is not None:
            self._to_manip_format(q0)
            default_q = q0
        else:
            # q0 = STRETCH_HOME_Q
            default_q = STRETCH_HOME_Q
        # Perform IK
        # These should be relative to the robot's base
        if relative:
            pos, quat = pose_query
        else:
            # We need to compute this relative to the robot...
            # So how do we do that?
            # This logic currently in local hello robot client
            raise NotImplementedError()

        q, success, debug_info = self.manip_ik_solver.compute_ik(
            pos, quat, q0, num_attempts=num_attempts, verbose=verbose
        )

        if q is not None and success:
            q = self._from_manip_format(q, default_q)
            self.set_config(q)

        return q, success, debug_info

    def get_ee_pose(self, q=None):
        if q is not None:
            self.set_config(q)
        return self.ref.get_link_pose(self._grasp_frame)

    def update_look_front(self, q):
        """look in front so we can see the floor"""
        return self.update_head(q, self.look_front)

    def update_look_ahead(self, q):
        """look straight ahead; cannot see terrain"""
        return self.update_head(q, self.look_ahead)

    def update_look_at_ee(self, q):
        """turn and look at ee area"""
        return self.update_head(q, self.look_at_ee)

    def interpolate_angle(self, qi, theta0, thetag, step=0.1):
        """just rotate to target angle"""
        if theta0 > thetag:
            thetag2 = thetag + 2 * np.pi
        else:
            thetag2 = thetag - 2 * np.pi
        dist1 = np.abs(thetag - theta0)
        dist2 = np.abs(thetag2 - theta0)
        # TODO remove debug code
        # print("....", qi)
        print("interp from", theta0, "to", thetag, "or maybe", thetag2)
        # print("dists =", dist1, dist2)
        if dist2 < dist1:
            dist = dist2
            thetag = thetag2
        else:
            dist = dist1
        # Dumb check to see if we can get the rotation right
        # if dist > np.pi:
        #    thetag -= np.pi/2
        #    dist = np.abs(thetag - theta0)
        # TODO remove debug code
        # print(" > interp from", theta0, "to", thetag)
        # Getting the direction right here
        dirn = 1.0 if thetag > theta0 else -1.0
        while dist > 0:
            qi = qi.copy()
            ai = np.zeros(self.dof)
            # TODO: we should handle look differently
            # qi = self.update_head(qi, self.look_front)
            if dist > step:
                dtheta = step
            else:
                dtheta = dist
            dist -= dtheta
            ai[2] = dirn * dtheta
            qi[HelloStretchIdx.BASE_THETA] += dirn * dtheta
            yield qi, ai

    def interpolate_arm(self, q0, qg, step=None):
        if step is None:
            step = self.default_step
        qi = q0
        while np.any(np.abs(qi - qg) > self.default_tols):
            qi = qi.copy()
            ai = qi.copy()
            ai[:3] = np.zeros(3)  # action does not move the base
            # TODO: we should handle look differently
            qi = self.update_head(qi.copy(), self.look_at_ee)
            dq = qg - qi
            dq = np.clip(dq, -1 * step, step)
            qi += dq
            qi = self.update_head(qi, self.look_at_ee)
            yield qi, ai

    def is_colliding(self, other):
        return self.ref.is_colliding(other)

    def extend_arm_to(self, q, arm):
        """
        Extend the arm by a certain amound
        Move the base as well to compensate.

        This is purely a helper function to make sure that we can find poses at which we can
        extend the arm in order to grasp.
        """
        a0 = q[HelloStretchIdx.ARM]
        a1 = arm
        q = q.copy()
        q[HelloStretchIdx.ARM] = a1
        theta = q[HelloStretchIdx.BASE_THETA] + np.pi / 2
        da = a1 - a0
        dx, dy = da * np.cos(theta), da * np.sin(theta)
        q[HelloStretchIdx.BASE_X] += dx
        q[HelloStretchIdx.BASE_Y] += dy
        return q

    def validate(self, q=None, ignored=[], distance=0.0, verbose=False):
        """
        Check collisions against different obstacles
        q = configuration to test
        ignored = other objects to NOT check against
        """
        self.set_config(q)
        # Check robot height
        if q[HelloStretchIdx.LIFT] >= 1.0:
            return False
        # Check links against obstacles
        for name, obj in self.backend.objects.items():
            if obj.id == self.ref.id:
                continue
            elif obj.id in ignored:
                continue
            elif self.ref.is_colliding(obj, distance=distance):
                if verbose:
                    print("colliding with", name)
                return False
        return True

    def create_action_from_config(self, q: np.ndarray) -> ContinuousFullBodyAction:
        """Create a default interface action from this"""
        xyt = np.zeros(3)
        xyt[0] = q[HelloStretchIdx.BASE_X]
        xyt[1] = q[HelloStretchIdx.BASE_Y]
        xyt[2] = q[HelloStretchIdx.BASE_THETA]
        return self.create_action(
            lift=q[HelloStretchIdx.LIFT],
            arm=q[HelloStretchIdx.ARM],
            pitch=q[HelloStretchIdx.WRIST_PITCH],
            roll=q[HelloStretchIdx.WRIST_ROLL],
            yaw=q[HelloStretchIdx.WRIST_YAW],
            xyt=xyt,
        )

    def create_action(
        self,
        lift=None,
        arm=None,
        roll=None,
        pitch=None,
        yaw=None,
        pan=None,
        tilt=None,
        xyt=None,
        defaults: np.ndarray = None,
    ) -> ContinuousFullBodyAction:
        """
        Original Arm Action Space: We define the action space that jointly controls (1) arm extension (horizontal), (2) arm height (vertical), (3) gripper wrist’s roll, pitch, and yaw, and (4) the camera’s yaw and pitch. The resulting size of the action space is 10.
        - Arm extension (size: 4): It consists of 4 motors that extend the arm: joint_arm_l0 (index 28 in robot interface), joint_arm_l1 (27), joint_arm_l2 (26), joint_arm_l3 (25)
        - Arm height (size: 1): It consists of 1 motor that moves the arm vertically: joint_lift (23)
        - Gripper wrist (size: 3): It consists of 3 motors that control the roll, pitch, and yaw of the gripper wrist: joint_wrist_yaw (31),  joint_wrist_pitch (39),  joint_wrist_roll (40)
        - Camera (size 2): It consists of 2 motors that control the yaw and pitch of the camera: joint_head_pan (7), joint_head_tilt (8)

        As a result, the original action space is the order of [joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3, joint_lift, joint_wrist_yaw, joint_wrist_pitch, joint_wrist_roll, joint_head_pan, joint_head_tilt] defined in habitat/robots/stretch_robot.py
        """
        assert self.joints_dof == 10
        if defaults is None:
            joints = np.zeros(self.joints_dof)
        else:
            assert len(defaults) == self.joints_dof
            joints = defaults.copy()
        if arm is not None:
            joints[:4] = np.ones(4) * (arm / 4.0)
        if lift is not None:
            joints[4] = lift
        if roll is not None:
            joints[5] = roll
        if pitch is not None:
            joints[6] = pitch
        if yaw is not None:
            joints[7] = yaw
        if pan is not None:
            joints[8] = pan
        if tilt is not None:
            joints[9] = tilt
        return ContinuousFullBodyAction(joints=joints, xyt=xyt)

    def delta_hab_to_position_command(self, cmd, pan, tilt, deltas) -> List:
        """Compute deltas"""
        assert len(deltas) == 10
        arm = deltas[0] + deltas[1] + deltas[2] + deltas[3]
        lift = deltas[4]
        roll = deltas[5]
        pitch = deltas[6]
        yaw = deltas[7]
        pan, tilt = self.head.get_pan_tilt()
        positions = [
            0,  # This is the robot's base x axis - not currently used
            cmd[1] + lift,
            cmd[2] + arm,
            cmd[3] + yaw,
            cmd[4] + pitch,
            cmd[5] + roll,
        ]
        pan = pan + deltas[8]
        tilt = tilt + deltas[9]
        return positions, pan, tilt

    def config_to_manip_command(self, q):
        """convert from general representation into arm manip command. This extracts just the information used for end-effector control: base x motion, arm lift, and wrist variables."""
        return [
            q[HelloStretchIdx.BASE_X],
            q[HelloStretchIdx.LIFT],
            q[HelloStretchIdx.ARM],
            q[HelloStretchIdx.WRIST_YAW],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_ROLL],
        ]

    def config_to_hab(self, q: np.ndarray) -> np.ndarray:
        """Convert default configuration into habitat commands. This is a slightly different format that strips out x, y, and theta."""
        hab = np.zeros(10)
        hab[0] = q[HelloStretchIdx.ARM]
        hab[4] = q[HelloStretchIdx.LIFT]
        hab[5] = q[HelloStretchIdx.WRIST_ROLL]
        hab[6] = q[HelloStretchIdx.WRIST_PITCH]
        hab[7] = q[HelloStretchIdx.WRIST_YAW]
        hab[8] = q[HelloStretchIdx.HEAD_PAN]
        hab[9] = q[HelloStretchIdx.HEAD_TILT]
        return hab

    def hab_to_position_command(self, hab_positions) -> List:
        """Compute hab_positions"""
        assert len(hab_positions) == 10
        arm = hab_positions[0] + hab_positions[1] + hab_positions[2] + hab_positions[3]
        lift = hab_positions[4]
        roll = hab_positions[5]
        pitch = hab_positions[6]
        yaw = hab_positions[7]
        positions = [
            0,  # This is the robot's base x axis - not currently used
            lift,
            arm,
            yaw,
            pitch,
            roll,
        ]
        pan = hab_positions[8]
        tilt = hab_positions[9]
        return positions, pan, tilt


if __name__ == "__main__":
    robot = HelloStretchKinematics()
    q0 = STRETCH_HOME_Q.copy()
    q1 = STRETCH_HOME_Q.copy()
    q0[2] = -1.18
    q1[2] = -1.1
    for state, action in robot.interpolate_angle(q0, q0[2], q1[2]):
        print(action)
