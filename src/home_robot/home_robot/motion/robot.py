# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import home_robot.utils.bullet as hrb


class Robot(object):
    """placeholder"""

    backends = ["bullet", "urdfpy"]

    def __init__(
        self,
        name="robot",
        urdf_path=None,
        visualize=False,
        assets_path=None,
        backend="bullet",
    ):

        self._backend = backend
        if self._backend == "bullet":
            # Load and create planner
            self.backend = hrb.PbClient(visualize=visualize)
            # Create object reference
            self.ref = self.backend.add_articulated_object(
                name, urdf_path, assets_path=assets_path
            )
        elif self._backend == "urdfpy":
            raise NotImplementedError()
        else:
            raise RuntimeError("backend not supported: " + str(backend))

    def get_backend(self):
        raise NotImplementedError

    def get_dof(self):
        """return degrees of freedom of the robot"""
        return self.dof

    def set_config(self, q):
        """put the robot in the right position"""
        raise NotImplementedError

    def get_config(self):
        """turn current state into a vector"""
        raise NotImplementedError

    def set_head_config(self, q):
        """just for the head"""
        raise NotImplementedError

    def set_camera_to_head(self, camera, q=None):
        """take a bullet camera and put it on the robot's head"""
        if q is not None:
            self.set_head_config(q)
        raise NotImplementedError
