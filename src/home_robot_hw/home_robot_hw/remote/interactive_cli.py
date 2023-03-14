# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Launches an interactive terminal with the Stretch API

import code

from .api import StretchClient

if __name__ == "__main__":
    robot = StretchClient()
    code.interact(local=locals())
