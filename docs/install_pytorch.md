
## Help Installing PyTorch

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


