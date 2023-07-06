## Configurations

### IPM

Under  `config.SLAP.IPM` for training:
- `validate (bool)`: Whether to run IPM in validation mode to visualize ground-truth as well
- `datadir (bool)`: Placeholder so user can input `--datadir <path/to/local/data` while running training or validation
- `template (bool)`: Collects all H5 files matching template `<path/to/local/data/temlate>`
- `load (string)`: Placeholder so user can pass which checkpoint to load for validation or resuming training
- `resume (bool)`: Load weights at `load` path before starting training
- `learning_rate (float)`: Learning rate for LAMB optimizer
- `task_name (string)`: Name of the task/experiment being trained, useful for wandb and directory tracking
- `source (string)`: Loads data using `stretch/franka' configuration
- `max_iter (int)`: Total number of training iterations
- `wandb (bool)`: Store run in wandb project
- `split (string)`: Placeholder, user can input `--split <path/to/train-test-split.yaml>` for consistent experiments (if not given anything the model trains on all samples collected from H5s found in `datadir`)
- `data_augmentation (bool)`: Train the model with positional, orientational, and cropping data-augmentation ON
- `color_jitter (bool)`: Use color-jitter for additional data-augmentation
- `loss_fn (string)`: Loss-function to use for training IPM classifier (supports xent and bce; xent gave a better result in experiments)
