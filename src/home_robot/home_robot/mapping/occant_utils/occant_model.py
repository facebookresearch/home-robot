import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from einops import rearrange

from home_robot.mapping.occant_utils.unet import (
    LearnedRGBProjection,
    MergeMultimodal,
    MiniUNetEncoder,
    ResNetRGBEncoder,
    UNetDecoder,
    UNetEncoder,
)


def softmax_2d(x):
    b, h, w = x.shape
    x_out = F.softmax(rearrange(x, "b h w -> b (h w)"), dim=1)
    x_out = rearrange(x_out, "b (h w) -> b h w", h=h)
    return x_out


# ================================ Anticipation base ==================================


class BaseModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "softmax":
            self.normalize_channel_0 = softmax_2d

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d

        self._create_gp_models()

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        x_dec_c0 = self.normalize_channel_0(x_dec[:, 0])
        x_dec_c1 = self.normalize_channel_1(x_dec[:, 1])
        return torch.stack([x_dec_c0, x_dec_c1], dim=1)


# ============================= Anticipation models ===================================


class ANSRGB(BaseModel):
    """
    Predicts depth-projection from RGB only.
    """

    def _create_gp_models(self):
        resnet = tmodels.resnet18(pretrained=True)
        self.main = nn.Sequential(  # (3, 128, 128)
            # Feature extraction
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,  # (512, 4, 4)
            # FC layers equivalent
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Upsampling
            nn.Conv2d(512, 256, 3, padding=1),  # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (256, 8, 8)
            nn.Conv2d(256, 128, 3, padding=1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (128, 16, 16),
            nn.Conv2d(128, 64, 3, padding=1),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (64, 32, 32),
            nn.Conv2d(64, 32, 3, padding=1),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (32, 64, 64),
            nn.Conv2d(32, 2, 3, padding=1),  # (2, 64, 64)
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # (2, 128, 128),
        )

    def _do_gp_anticipation(self, x):
        x_dec = self.main(x["rgb"])
        x_dec = self._normalize_decoder_output(x_dec)
        outputs = {"occ_estimate": x_dec}

        return outputs


class ANSDepth(BaseModel):
    """
    Computes depth-projection from depth and camera parameters only.
    Outputs the GT projected occupancy
    """

    def _create_gp_models(self):
        pass

    def _do_gp_anticipation(self, x):
        x_dec = x["ego_map_gt"]
        outputs = {"occ_estimate": x_dec}

        return outputs


class OccAntRGB(BaseModel):
    """
    Anticipated using rgb only.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth projection estimator
        config = self.config.clone()
        self.gp_depth_proj_estimator = ANSRGB(config)

        # Depth encoder branch
        self.gp_depth_proj_encoder = UNetEncoder(2, nsf=nsf)

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)

        self._detach_depth_proj = gp_cfg.detach_depth_proj

        # Load pretrained model if available
        if gp_cfg.pretrained_depth_proj_model != "":
            self._load_pretrained_model(gp_cfg.pretrained_depth_proj_model)

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False
        if gp_cfg.freeze_depth_proj_model:
            for p in self.gp_depth_proj_estimator.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'depth' - (bs, 3, H, W) Depth input - channels are repeated
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, 768, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, 768, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        # Estimate projected occupancy
        x_depth_proj = self.gp_depth_proj_estimator(x)["occ_estimate"]  # (bs, 2, V, V)
        if self._detach_depth_proj:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj.detach()
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}
        else:
            x_depth_proj_enc = self.gp_depth_proj_encoder(
                x_depth_proj
            )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)
        x_dec = self._normalize_decoder_output(x_dec)  # (bs, 2, H, W)

        outputs = {"depth_proj_estimate": x_depth_proj, "occ_estimate": x_dec}

        return outputs

    def _load_pretrained_model(self, path):
        depth_proj_state_dict = torch.load(
            self.config.GP_ANTICIPATION.pretrained_depth_proj_model, map_location="cpu"
        )["mapper_state_dict"]
        cleaned_state_dict = {}
        for k, v in depth_proj_state_dict.items():
            if ("mapper_copy" in k) or ("projection_unit" not in k):
                continue
            new_k = k.replace("module.", "")
            new_k = new_k.replace("mapper.projection_unit.main.main.", "")
            cleaned_state_dict[new_k] = v
        self.gp_depth_proj_estimator.load_state_dict(cleaned_state_dict)


class OccAntDepth(BaseModel):
    """
    Anticipated using depth projection only.
    """

    def _create_gp_models(self):
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(2, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        self.gp_depth_proj_encoder = unet_encoder
        self.gp_decoder = unet_decoder

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'ego_map_gt' - (bs, 2, H, W) input
        """
        x_enc = self.gp_depth_proj_encoder(
            x["ego_map_gt"]
        )  # dictionary with different outputs
        x_dec = self.gp_decoder(x_enc)  # (bs, 2, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        return outputs


class OccAntRGBD(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """

    def _create_gp_models(self):
        nmodes = 2
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf
        unet_encoder = UNetEncoder(2, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth encoder branch
        self.gp_depth_proj_encoder = unet_encoder

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = unet_decoder

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, infeats, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, infeats, H/4, W/4)

        x_rgb_enc = self.gp_rgb_unet(x_gp)  # {'x3p', 'x4p', 'x5p'}
        x_depth_proj_enc = self.gp_depth_proj_encoder(
            x["ego_map_gt"]
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16)
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 )
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 )
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)  # (bs, 2, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        return outputs


class OccAntGroundTruth(BaseModel):
    """
    Outputs the GT anticipated occupancy
    """

    def _create_gp_models(self):
        pass

    def _do_gp_anticipation(self, x):
        x_dec = x["ego_map_gt_anticipated"]  # (bs, 2, H, W)
        outputs = {"occ_estimate": x_dec}

        return outputs


# ================================ Occupancy anticipator ==============================


class OccupancyAnticipator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        model_type = cfg.type
        self._model_type = model_type
        cfg.defrost()
        if model_type == "ans_rgb":
            self.main = ANSRGB(cfg)
        elif model_type == "ans_depth":
            self.main = ANSDepth(cfg)
        elif model_type == "occant_rgb":
            self.main = OccAntRGB(cfg)
        elif model_type == "occant_depth":
            self.main = OccAntDepth(cfg)
        elif model_type == "occant_rgbd":
            self.main = OccAntRGBD(cfg)
        elif model_type == "occant_ground_truth":
            self.main = OccAntGroundTruth(cfg)
        else:
            raise ValueError(f"Invalid model_type {model_type}")

        cfg.freeze()

    def forward(self, x):
        return self.main(x)

    @property
    def use_gp_anticipation(self):
        return self.main.use_gp_anticipation

    @property
    def model_type(self):
        return self._model_type
