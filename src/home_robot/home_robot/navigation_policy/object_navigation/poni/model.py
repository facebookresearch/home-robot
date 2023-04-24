import torch.nn as nn

from home_robot.mapping.semantic.constants import MapConstants as MC

from .utils import get_activation_fn, get_semantic_encoder_decoder


class PFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define models
        enable_area_head = self.cfg.DATASET.enable_unexp_area
        if enable_area_head:
            self.cfg.defrost()
            self.cfg.MODEL.enable_area_head = enable_area_head
            self.cfg.freeze()
        (
            self.encoder,
            self.object_decoder,
            self.area_decoder,
        ) = get_semantic_encoder_decoder(self.cfg)
        # Define activation functions
        self.object_activation = get_activation_fn(self.cfg.MODEL.object_activation)
        self.area_activation = get_activation_fn(self.cfg.MODEL.area_activation)

    def forward(self, x):
        embedding = self.encoder(x)
        object_preds = self.object_activation(self.object_decoder(embedding))
        area_preds = None
        if self.area_decoder is not None:
            area_preds = self.area_activation(self.area_decoder(embedding))
        return object_preds, area_preds

    def infer(self, x):
        object_preds, area_preds = self(x)
        return object_preds, area_preds
