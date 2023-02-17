# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "home-robot",
    "habitat-sim",
    # Installing habitat-lab directly from git (does not work due to bug in habitat-lab@v0.2.2):
    # "habitat @ git+ssh://git@github.com/facebookresearch/habitat-lab.git@v0.2.2",
    # Latest habitat version (not used due to Hydra migration):
    # "habitat-lab @ git+ssh://git@github.com/facebookresearch/habitat-lab@stable&subdirectory=habitat-lab",
]

setup(
    name="home-robot-sim",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
