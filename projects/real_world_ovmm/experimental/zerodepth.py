# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# To get this to work:
#  pip install torch-scatter einops lpip


import matplotlib.pyplot as plt
import torch

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.remote.api import StretchClient

zerodepth_model = torch.hub.load(
    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
)
intrinsics = torch.zeros((1, 3, 3))
rgb = torch.zeros((1, 3, 256, 256))
depth_pred = zerodepth_model(rgb, intrinsics)

client = StretchClient()

rgb, depth, xyz = client.head.get_images(compute_xyz=True)
plt.imshow(rgb)
plt.show()

rgb = torch.FloatTensor(rgb[None]).permute(0, 3, 1, 2)
intrinsics = torch.FloatTensor(client.head.intrinsics()[None])
pred_depth = zerodepth_model(rgb, intrinsics)[0, 0].detach().cpu().numpy()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(depth)
plt.subplot(1, 2, 2)
plt.imshow(pred_depth)

breakpoint()
