# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from .modules.models.lseg_net import LSegEncDecNet


def load_lseg_for_inference(
    checkpoint_path: str, device: torch.device, visualize=True
) -> LSegEncDecNet:
    model = LSegEncDecNet(
        arch_option=0, block_depth=0, activation="lrelu", visualize=visualize
    )

    model_state_dict = model.state_dict()
    pretrained_state_dict = torch.load(checkpoint_path)
    pretrained_state_dict = {
        k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
    }
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.to(device)
    return model
