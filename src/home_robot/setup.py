# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup

install_requires = [
    "numpy",
]

# For data_tools sub-module
data_tools_requires = ["h5py", "imageio", "pygifsicle"]
install_requires += data_tools_requires

setup(
    name="home-robot",
    version="0.1.0",
    packages=["home_robot"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
