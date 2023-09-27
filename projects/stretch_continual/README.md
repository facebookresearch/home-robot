# Evaluating Continual Learning on a Home Robot
This project contains the environment code for running continual imitation learning on a Stretch robot, as used in
[Evaluating Continual Learning on a Home Robot](https://arxiv.org/abs/2306.02413) (CoLLAs, 2023).

In particular, it provides the ability to:
1. Collect demonstrations (`stretch_collect_demo_env.py`)
1. Use existing demonstrations (`stretch_offline_demo_env.py`)
1. Run a policy on the real robot hardware (`stretch_live_env.py`). 

These environments adhere to the OpenAI gym API, so they can be more readily used with, for example, reinforcement learning 
methods and frameworks.

Please see [continual_rl](https://github.com/AGI-Labs/continual_rl) for the code for the methods presented in the above paper,
as well as the framework ([CORA](https://arxiv.org/abs/2110.10067)) in which it was run.

## Citation
If you use this work, please cite it with:

```bibtex
@misc{powers2023evaluating,
      title={Evaluating Continual Learning on a Home Robot}, 
      author={Sam Powers and Abhinav Gupta and Chris Paxton},
      year={2023},
      eprint={2306.02413},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

Note: accepted to CoLLAs 2023; a PMLR citation will be available after publication.