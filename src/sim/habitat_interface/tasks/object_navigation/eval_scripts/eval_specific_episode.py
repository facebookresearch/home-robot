"""
This script is intended to run from the "src" root:
python sim/habitat_interface/tasks/object_navigation/eval_scripts/eval_specific_episode.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from habitat.core.env import Env
from habitat.core.simulator import Observations

from sim.habitat_interface.utils.config_utils import get_config
from sim.habitat_interface.tasks.object_navigation.agent.objectnav_agent import ObjectNavAgent


def reset_to_episode(env: Env,
                     scene_id: str,
                     episode_id: str) -> Observations:
    """
    Adapted from:
    https://github.com/facebookresearch/habitat-lab/blob/main/habitat/core/env.py
    """
    env._reset_stats()

    episode = [e for e in env.episodes
               if e.episode_id == episode_id and
               e.scene_id.endswith(f"{scene_id}.basis.glb")][0]
    env._current_episode = episode

    env._episode_from_iter_on_reset = True
    env._episode_force_changed = False

    env.reconfigure(env._config)

    observations = env.task.reset(episode=env.current_episode)
    env._task.measurements.reset_measures(
        episode=env.current_episode,
        task=env.task,
        observations=observations,
    )
    return observations


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "configs/agent/hm3d_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.freeze()

    agent = ObjectNavAgent(config=config)
    env = Env(config=config.TASK_CONFIG)

    scene_id = "ziup5kvtCCR"
    episode_id = "1"

    obs = reset_to_episode(env, scene_id, episode_id)
    agent.reset()
    agent.set_vis_dir(scene_id=scene_id, episode_id=episode_id)

    t = 0
    while not env.episode_over:
        t += 1
        print(t)
        print(obs.keys())
        action = agent.act(obs)
        obs = env.step(action)

    print(env.get_metrics())
