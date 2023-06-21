mkdir -p data/datasets/objectnav/hm3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip -O data/datasets/objectnav/hm3d/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
