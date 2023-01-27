# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

def fk_ik_helper(rob, q):
    """ do (1) generate random robot 
        generate fk for it; randomize robot
        """
    pass

def test_fk_ik():
    rob = HelloStretch()
    for i in range(1000):
        res = fk_ik_helper(rob, q)
