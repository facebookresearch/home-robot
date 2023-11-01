# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "empy",
    "catkin_pkg",
    "rospkg",
    "rosnumpy",
]

setup(
    name="home_robot_hw",
    version="1.0.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
