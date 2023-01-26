# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup

install_requires = ["numpy", "empy", "catkin_pkg", "rospkg"]

setup(
    name="home_robot",
    version="1.0.0",
    packages=["home_robot"],
    package_dir={"": ".."},
    install_requires=install_requires,
    zip_safe=False,
)
