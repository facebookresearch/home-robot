# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "numpy==1.23",
    "scipy>=1.11",
    "hydra-core",
    "yacs",
    "h5py>3.5",
    "pybullet",
    "pygifsicle",
    "open3d==0.17",
    "numpy-quaternion>=2022.4",
    "pybind11-global",
    "sophuspy",
    "trimesh>4",
    "pin>=2.6.17",
    "torch_cluster",
    "torch_scatter",
    "pillow==9.5.0",  # For Detic compatibility
]

setup(
    name="home-robot",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
