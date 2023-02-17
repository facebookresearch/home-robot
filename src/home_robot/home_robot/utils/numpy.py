# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime

import numpy as np


def to_npy_file(description="data", **data):
    """write all these different data into a file. will be of the structure:
    description_datetime.npy
    """
    now = datetime.now()
    datestr = now.strftime("%Y_%m_%d-%H_%M_%S")
    filename = "%s_%s.npy" % (description, datestr)

    # This is for
    np.save(filename, data)


def load_npy_file(filename):
    """assume top level is a dictionary"""
    return np.load(filename, allow_pickle=True)[()]
