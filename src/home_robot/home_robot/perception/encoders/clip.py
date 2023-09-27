# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import clip
import numpy as np
import torch
from PIL import Image


class ClipEncoder:
    """Simple wrapper for encoding different things as text."""

    def __init__(self, version="ViT-B/32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.version = version
        self.model, self.preprocess = clip.load(self.version, device=self.device)

    def encode_image(self, image: np.ndarray):
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        return image_features

    def encode_text(self, text: str):
        """Return clip vector for text"""
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features
