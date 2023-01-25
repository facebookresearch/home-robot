# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup

install_requires = [
    "numpy",
    "Pyro5",
    "pynput",
]

setup(
    name="home-robot-client",
    version="0.1.0",
    packages=["home_robot_client"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
