# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup

install_requires = [
    "numpy",
    "home-robot",
    # f"habitat-lab @ {os.path.dirname(os.path.realpath(__file__))}/third_party/habitat-lab/habitat-lab",
]

setup(
    name="home-robot-sim",
    version="0.1.0",
    packages=["home_robot_sim"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
