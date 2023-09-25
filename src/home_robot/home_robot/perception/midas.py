# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from midas.model_loader import default_models, load_model
from midas.run import process


class Midas:
    def __init__(self, device):

        super().__init__()
        # midas params
        self.device = device
        self.model_type = "dpt_beit_large_512"
        self.optimize = False
        height = None
        square = False
        model_path = f"src/third_party/MiDaS/weights/{self.model_type}.pt"
        self.model, self.transform, self.net_w, self.net_h = load_model(
            device, model_path, self.model_type, self.optimize, height, square
        )

    # expects numpy rgb, [0,255]
    # TODO: undefined name "process"
    def depth_estimate(self, rgb: np.ndarray, depth: np.ndarray):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        image = self.transform({"image": (rgb / 255)})["image"]
        # compute
        with torch.no_grad():
            prediction = process(
                self.device,
                self.model,
                self.model_type,
                image,
                (self.net_w, self.net_h),
                rgb.shape[1::-1],
                self.optimize,
                False,
            )
        depth_valid = depth > 0

        # solve for MSE for the system of equations Ax = b where b is the observed depth and x is the predicted depth values
        x = np.stack(
            (prediction[depth_valid], np.ones_like(prediction[depth_valid])), axis=1
        ).T
        b = depth[depth_valid].T
        # 1 x 2 * 2 x n = 1 x n
        pinvx = np.linalg.pinv(x)
        A = b @ pinvx

        adjusted = prediction * A[0] + A[1]
        mse = ((A @ x - b) ** 2).mean()
        mean_error = np.abs(A @ x - b).mean()
        return adjusted, mse, mean_error
