from typing import Tuple, Optional
import json
import yaml
import yacs
from yacs.config import CfgNode as CN


# def make_config_recursive(entries):
#     new_entries = {}
#     for k, v in entries.items():
#         if isinstance(v, dict):
#             entries[k] = make_config_recursive(v)

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(path: str, opts: Optional[list] = None) -> Tuple[CN, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """

    # Start with our code's config
    print("Loading config from:", path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        config = Config(**data)
    breakpoint()
    #$ config.merge_from_file(path)

    # Add command line arguments
    if opts is not None:
        raise NotImplementedError()
    #    config.merge_from_list(opts)
    # config.freeze()

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
