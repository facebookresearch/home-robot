from yacs.config import CfgNode as CN

_C = CN()
_C.type = "occant_depth"
# entropy threshold to classify a cell as confident
_C.thresh_entropy = 0.5
# threshold to classify a cell as explored
_C.thresh_explored = 0.6
# threshold to classify a cell as obstacle
_C.thresh_obstacle = 0.6
_C.input_hw = [128, 128]
_C.image_mean = [0.485, 0.456, 0.406]
_C.image_std = [0.229, 0.224, 0.225]
#####################################################################
# Anticipation model
#####################################################################
_C.GP_ANTICIPATION = CN()
# Model capacity factor for custom UNet
_C.GP_ANTICIPATION.unet_nsf = 16
# Freeze image features?
_C.GP_ANTICIPATION.freeze_features = False
_C.GP_ANTICIPATION.nclasses = 2
_C.GP_ANTICIPATION.resnet_type = "resnet18"
# OccAnt RGB specific hyperparameters
_C.GP_ANTICIPATION.detach_depth_proj = False
_C.GP_ANTICIPATION.pretrained_depth_proj_model = ""
_C.GP_ANTICIPATION.freeze_depth_proj_model = False
# Normalization options for anticipation output
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION = CN()
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 = "sigmoid"
_C.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 = "sigmoid"

#####################################################################
# Map aggregation
#####################################################################
_C.AGGREGATOR = CN()
_C.AGGREGATOR.registration_type = "moving_average"
_C.AGGREGATOR.map_registration_momentum = 0.9


def get_cfg(path=None):
    cfg = _C.clone()
    cfg.defrost()
    if path is not None:
        cfg.merge_from_file(path)
    cfg.freeze()
    return cfg
