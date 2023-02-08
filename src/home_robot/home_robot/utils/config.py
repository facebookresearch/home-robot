from typing import Tuple, Optional
import json
import yaml
import yacs.config


class Config(yacs.config.CfgNode):
    """ store a yaml config """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


def get_config(path: str, opts: Optional[list] = None) -> Tuple[Config, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """

    # Start with our code's config
    config = Config()
    config.merge_from_file("configs/agent/floorplanner_eval.yaml")

    # Add command line arguments
    if opts is not None:
        config.merge_from_list(opts)
    config.freeze()

    # Generate a string representation of our code's config
    config_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    if opts is not None:
        for i in range(0, len(opts), 2):
            dict = config_dict
            keys = opts[i].split(".")
            if "TASK_CONFIG" in keys:
                continue
            value = opts[i + 1]
            for key in keys[:-1]:
                dict = dict[key]
            dict[keys[-1]] = value
    config_str = json.dumps(config_dict, indent=4)

    return config, config_str
