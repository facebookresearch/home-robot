# Habitat OVMM

## Dataset Setup

### Scene dataset setup 

```
cd `HOME_ROBOT_ROOT/data/`
# Download the scenes
git clone https://huggingface.co/datasets/osmm/fpss --branch osmm
# Download the objects and metadata
git clone https://huggingface.co/datasets/osmm/objects
```

### Other instructions

Rough notes; some things were missing for configuring a new environment:
  - Download the objects into `HOME_ROBOT_ROOT/data`
  - Download the urdf into `HOME_ROBOT_ROOT/data/robots/hab_stretch` - robot should be at `data/robots/hab_stretch/urdf/` - robot from [the FAIR distribution here in zip format](http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip)


### Episode dataset setup
```

cd `HOME_ROBOT_ROOT/data/`
git clone https://huggingface.co/datasets/osmm/episodes
```

## Demo setup

Run
```
python projects/habitat_ovmm/eval_dataset.py
```

Results are saved to `datadump/images/eval_floorplanner/`.


## Run

> Note: Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/hssd_eval.yaml` to test DETIC perception.

```
cd /path/to/home-robot


# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_dataset.py

# Evaluation on complete episode dataset with DETIC
python projects/habitat_ovmm/eval_dataset.py  --baseline_config_path projects/habitat_ovmm/configs/agent/floorplanner_detic_eval.yaml

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_dataset.py habitat.dataset.episode_ids="[151,182]"
```
