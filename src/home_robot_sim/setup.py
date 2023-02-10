# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup

install_requires = [
    "numpy",
    "home-robot",
    "habitat-sim",
    "habitat @ git+ssh://git@github.com/facebookresearch/habitat-lab.git@v0.2.2",
    # "habitat-lab @ git+ssh://git@github.com/facebookresearch/habitat-lab@stable&subdirectory=habitat-lab",
]

setup(
    name="home-robot-sim",
    version="0.1.0",
    packages=["home_robot_sim"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
