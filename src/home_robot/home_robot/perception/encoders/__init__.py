# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any

from .clip_encoder import ClipEncoder


def get_encoder(encoder_name, args: Any):
    if encoder_name == "clip":
        return ClipEncoder(args)
    elif encoder_name == "mtm":
        from .mtm_encoder import HomeRobotMTMEncoder

        return HomeRobotMTMEncoder()
