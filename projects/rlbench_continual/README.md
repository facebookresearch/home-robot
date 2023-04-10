# Installation

## Base installation
1. Follow the regular instructions to install RLBench: https://github.com/stepjam/RLBench
1. If you have sudo access, you can follow their instructions as-given to install headless. See below for headless without sudo
1. Install torch_geometric and its dependencies: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Headless without sudo access (Ubuntu) -- TODO: testing in progress:
1. Download the dpkg without installing it: `apt-get download xvfb`
1. Extract the dpkg to some location (here `~/xvfb`), again without install: `dpkg -x xvfb[version info].deb ~/xvfb`
1. Start xvfb and assign it to a screen number (here `:1`): `~/xvfb/usr/bin/Xvfb :1 -screen 0 1024x768x16 &`
    1. TODO: probably can `xvfb-run` instead, if that's your preference (get directions for this)
1. Run experiments with the `DISPLAY=:1` environment variable