# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from home_robot.utils.config import get_config


def load_config(visualize: bool = False, print_images: bool = True, config_path=None):
    """Load config path for real world experiments and use proper presets."""
    if config_path is None:
        config_path = "projects/real_world_ovmm/configs/agent/eval.yaml"
    config, _ = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    if config.GROUND_TRUTH_SEMANTICS != 0:
        raise RuntimeError("No ground truth semantics in the real world!")
    config.freeze()
    return config


def load_slap_config(
    visualize: bool = False, print_images: bool = True, config_path=None
):
    """Load config path for real world experiments and use proper presets."""
    config, _ = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    if config.GROUND_TRUTH_SEMANTICS != 0:
        raise RuntimeError("No ground truth semantics in the real world!")
    config.freeze()
    return config
