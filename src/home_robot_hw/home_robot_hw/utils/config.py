from home_robot.utils.config import get_config


def load_config(
    visualize: bool = False, print_images: bool = True, config_path=None, **kwargs
):
    """Load config path for real world experiments and use proper presets."""
    if config_path is None:
        # TODO: make sure this is the right default
        config_path = "projects/stretch_ovmm/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    config.freeze()
    return config
