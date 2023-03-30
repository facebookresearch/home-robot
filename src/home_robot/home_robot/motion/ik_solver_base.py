from typing import Optional, Tuple

import numpy as np


class IKSolverBase(object):
    """
    Base class for all IK solvers.
    """

    def get_dof(self) -> int:
        """returns dof for the manipulation chain"""
        raise NotImplementedError()

    def get_num_controllable_joints(self) -> int:
        """returns number of controllable joints under this solver's purview"""
        raise NotImplementedError()

    def compute_fk(self, q) -> Tuple[np.ndarray, np.ndarray]:
        """given joint values return end-effector position and quaternion associated with it"""
        raise NotImplementedError()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        num_attempts: int = 1,
        verbose: int = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Given an end-effector position and quaternion, return the joint states and a success flag.
        Some solvers (e.g. the PositionIKOptimizer solver) will return a result regardless; the success flag indicates
        if the solution is within the solver's expected error margins.
        """
        raise NotImplementedError()
