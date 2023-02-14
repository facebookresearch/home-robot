# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup

install_requires = ["numpy<1.24", "scipy", "sophuspy", "hydra-core"]

setup(
    name="home-robot",
    version="0.1.0",
    packages=["home_robot"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
