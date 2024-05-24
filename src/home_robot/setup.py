# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

install_requires = [
    "mkl==2024.0",
    "numpy<1.24",
    "scipy==1.13.0",
    "hydra-core",
    "yacs",
    "h5py==3.11.0",
    "pybullet",
    "pygifsicle",
    "open3d",
    "numpy-quaternion",
    "pybind11-global",
    "sophuspy==0.0.8",
    "trimesh==3.23.5",
    "pin>=2.6.17",
    "torch_cluster",
    "torch_scatter",
    "pillow==10.3.0",  # For Detic compatibility
]

setup(
    name="home-robot",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=install_requires,
    include_package_data=True,
)
