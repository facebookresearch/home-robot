# Simulation backend for the Hello Stretch

- WIP
- Contains a minimal, dummy simulation for (debugging & automated testing), to be replaced with Habitat simulation
- Habitat simulation: wrapper around [habitat-sim](https://github.com/facebookresearch/habitat-sim), not yet implemented
- Same interface as home_robot_hw

## Installation

After installing [home_robot](../home_robot):

```sh
# Install simulation deps
mamba install -c conda-forge -c aihabitat habitat-sim withbullet
git submodule update --init
pip install -e src/third_party/habitat-lab  # NOTE: Habitat-lab@v0.2.2 only works in editable mode

## Demo setup

Run
```
python projects/habitat_ovmm/eval_vectorized.py
```

Results are saved to `datadump/images/eval_floorplanner/`.

## Install Detic

```sh
git submodule update --init --recursive src/third_party/detectron2 src/home_robot/home_robot/perception/detection/detic/Detic
pip install -e src/third_party/detectron2

cd src/home_robot/home_robot/perception/detection/detic/Detic
pip install -r requirements.txt

mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Test it with
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

cd -
```

## Run

> Note: Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/floorplanner_eval.yaml` to test DETIC perception.

```
cd /path/to/home-robot


# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_vectorized.py

# Evaluation on complete episode dataset with DETIC
python projects/habitat_ovmm/eval_vectorized.py  --baseline_config_path projects/habitat_ovmm/configs/agent/floorplanner_detic_eval.yaml

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_vectorized.py habitat.dataset.episode_ids="[151,182]"
```
