# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import time

import yaml


def gen_all_configs(base_config_path: str, save_folder: str) -> None:
    """
    Generates possible configs for all variants of the baseline
    """
    base_config = yaml.safe_load(open(base_config_path, "r"))
    timestamp = time.strftime("%m%d-%h%m%s")
    os.makedirs(save_folder, exist_ok=True)
    # loop over all possible choices of: viz/no_viz, heuristic/rl nav, heuristic/rl manip, GT/DETIC perception
    for viz in ["viz", "no_viz"]:
        for manip in ["heuristic", "rl"]:
            for nav in ["heuristic", "rl"]:
                for perception in ["gt", "detic"]:
                    config = base_config.copy()
                    config["EXP_NAME"] = (
                        f"ovmm_{timestamp}/"
                        + manip[0]
                        + "_m_"
                        + nav[0]
                        + "_n_"
                        + perception
                        + (f"_{viz}" if viz != "no_viz" else "")
                    )
                    config["AGENT"]["SKILLS"]["NAV_TO_OBJ"]["type"] = nav
                    config["AGENT"]["SKILLS"]["NAV_TO_REC"]["type"] = nav
                    config["AGENT"]["SKILLS"]["GAZE_OBJ"]["type"] = manip
                    config["AGENT"]["SKILLS"]["PLACE"]["type"] = manip

                    config["AGENT"]["skip_skills"]["nav_to_obj"] = False
                    config["AGENT"]["skip_skills"]["nav_to_rec"] = False
                    config["AGENT"]["skip_skills"]["gaze_at_obj"] = manip != "rl"
                    config["AGENT"]["skip_skills"]["gaze_at_rec"] = True
                    config["AGENT"]["skip_skills"]["pick"] = False
                    config["AGENT"]["skip_skills"]["place"] = False

                    if viz != "no_viz":
                        config["PRINT_IMAGES"] = 1
                        config["AGENT"]["SKILLS"]["PICK"]["type"] = "heuristic"
                    else:
                        config["PRINT_IMAGES"] = 0
                        config["AGENT"]["SKILLS"]["PICK"]["type"] = "oracle"

                    config["GROUND_TRUTH_SEMANTICS"] = 1 if perception == "gt" else 0

                    # Write out config
                    config_path = (
                        save_folder
                        + "/"
                        + manip[0]
                        + "_m_"
                        + nav[0]
                        + "_n_"
                        + perception
                        + ".yaml"
                    )
                    if viz != "no_viz":
                        config_path = config_path.replace(".yaml", f"_{viz}.yaml")

                    with open(config_path, "w") as f:
                        f.write("###### AUTOMATICALLY GENERATED. DO NOT EDIT. ######\n")
                        yaml.dump(config, f)


if __name__ == "__main__":
    # define arguments for reading from command line
    parser = argparse.ArgumentParser(
        description="Generating configs automatically for baseline variants"
    )
    parser.add_argument(
        "--base-config-path",
        type=str,
        help="path to base config",
        default="projects/habitat_ovmm/configs/agent/hssd_eval.yaml",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="path to store the configs",
        default=os.path.join(
            "projects/habitat_ovmm/configs/agent/generated",
            time.strftime("%m%d-%H%M%S"),
        ),
    )

    args = parser.parse_args()
    gen_all_configs(args.base_config_path, args.save_folder)
