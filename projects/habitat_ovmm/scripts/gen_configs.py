import os
import time

import yaml

base_config_path = "projects/habitat_ovmm/configs/agent/hssd_eval.yaml"

# Read in base config
base_config = yaml.safe_load(open(base_config_path, "r"))

# get date
timestamp = time.strftime("%m%d-%H%M%S")
folder_name = os.path.join("projects/habitat_ovmm/configs/agent/generated", timestamp)
os.makedirs(folder_name, exist_ok=True)

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
                    folder_name
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
                    yaml.dump(config, f)
