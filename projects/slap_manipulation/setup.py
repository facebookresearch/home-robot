# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

__author__ = "Priyam Parashar"
__copyright__ = "2023, Meta"


setup(
    name="slap_manipulation",
    author="Priyam Parashar",
    author_email="priparashar@meta.com",
    version="0.3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # scripts=["scripts/collect_data_and_cal.py", "scripts/record_calibration_points.py"],
)
