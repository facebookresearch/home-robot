
## Debugging Common issues

You might see some problems with PyTorch and other python code installation. This document gives some advice.

### Make sure cuda and torch versions match

Make sure that cuda and pytorch versions are the same in your installation:

```
~/src/home-robot$ conda list | grep torch
ffmpeg                    4.3                  hf484d3e_0    pytorch
pyg                       2.3.1           py39_torch_1.13.0_cu117    pyg
pytorch                   1.13.1          py3.9_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cluster           1.6.1           py39_torch_1.13.0_cu117    pyg
pytorch-cuda              11.7                 h778d358_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytorch-scatter           2.1.1           py39_torch_1.13.0_cu117    pyg
pytorch3d                 0.7.4           py39_cu117_pyt1131    pytorch3d
torchaudio                0.13.1               py39_cu117    pytorch
torchvision               0.14.1               py39_cu117    pytorchs
```

The numbers after `cu` should be the same, i.e. `cu117`.

This should be true even after installing ROS noetic, e.g. with robostack:
```
# Installs ROS via Rboostack
mamba env create -n home-robot -f src/home_robot_hw/environment.yml
```

## Help Installing PyTorch

You might see some problems with installing 

### Installing Pytorch

See [here](https://pytorch.org/get-started/locally/) to install PyTorch. Example command:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### PyTorch3d

To install PyTorch3d, run:

```
conda install pytorch3d -c pytorch3d
```

If this causes trouble, building from source works reliably, but you must make sure you have the correct CUDA version on your workstation:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

See the [PyTorch3d installation page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more information.


