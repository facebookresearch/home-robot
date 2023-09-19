# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import sophus as sp


@dataclass
class ManipulatorBaseParams:
    se3_base: sp.SE3
