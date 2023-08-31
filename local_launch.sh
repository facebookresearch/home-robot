# TODO: Profile
# TODO: depth point from point cloud
rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/map_visualization/video.mp4 ~/data/vis/dynamic_map.mp4
rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/program.prof ~/data/program.prof
rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/
cd /home/akshara/asc_demo/home-robot/projects/spot/
rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/fremont_trajectories/bear_place ~/data/fremont_trajectories

rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/fremont_trajectories/spot_abnb1_trial9_test2 ~/data/fremont_trajectories

rsync -vaz akshara:/home/akshara/asc_demo/home-robot/projects/spot/fremont_trajectories/spot_abnb1_video6 ~/data/fremont_trajectories

rsync -vazu ~/data/fremont_trajectories devfair:/private/home/matthewchang/spot_trajectories
rsync -vaz akshara-spot:/home/akshara/Desktop/vids_to_copy ~/data/vis/

cd /home/akshara/asc_demo/home-robot
cd /home/kavit/fair/home-robot
bash
DISPLAY=:1 python projects/spot/objectnav.py --category 'teddy bear' --rotate
DISPLAY=:1 python projects/spot/objectnav.py --category 'teddy bear' --keyboard
DISPLAY=:1 python projects/spot/goat.py
DISPLAY=:1 python projects/spot/reset.py

#indexing into the map as [a,b] gives a = +y, b = +x in the visualizations, the agent starts facing positive x
# i.e. it's row column and the visiualization flips vertically
#
DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory15 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6 --keyboard
DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory15 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6
DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory16 --goals=image_refrigerator1,object_toilet

DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory16 --goals=image_refrigerator1,object_toilet

DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory16 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6

DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory17 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6
DISPLAY=:1 python projects/spot/goat.py --trajectory=test --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6 --keyboard
DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory19 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6



rsync -vaz akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/fremont_trajectories/trajectory16 ~/data/


DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory18 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6 --offline

DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory20 --goals=object_toilet,image_bed1,image_chair5,language_couch1,object_book,image_chair6 

DISPLAY=:1 python -m cProfile -o program.prof projects/spot/goat.py --trajectory=trajectory21 --goals=object_toilet,image_bed1,image_refrigerator1,language_couch1,object_book,image_chair6 
DISPLAY=:1 python projects/spot/goat.py --trajectory=trajectory21 --goals=object_bear,object_couch1 --pick-place

DISPLAY=:1 python projects/spot/goat.py --trajectory=chair_test --goals=object_chair, --pick-place

python projects/spot/goat.py --trajectory=~/data/fremont_trajectories/chair_test --goals=object_chair, --pick-place --offline

DISPLAY=:1 python projects/spot/goat.py --trajectory=bear_pick_place --goals=object_bear,object_bed --pick-place

DISPLAY=:1 python projects/spot/goat.py --trajectory=bear_pick_place --goals=object_bear,object_bed --pick-place --offline

DISPLAY=:1 python projects/spot/goat.py --trajectory=bear_place --goals=place_object_bed


# for sophuspy
brew install pybind11
pip install scipy sophuspy pybullet trimesh pytest scikit-image scikit-fmm scikit-learn numpy-quaternion natsort timm pandas matplotlib click yacs h5py imageio pygifsicle pynput git+https://github.com/openai/CLIP.git 
pip install transforms3d gym blosc imutils

git submodule update --init src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
git submodule update --init src/third_party/MiDaS
pip install -e src/third_party/MiDaS


__________ IRL experiments

python resize_image.py /Users/matthewchang/Desktop/airbnb_images/airbnb1
rsync -vaz /Users/matthewchang/Desktop/airbnb_images/airbnb1_resized/ akshara-spot:/home/akshara/asc_demo/home-robot/projects/spot/airbnb1_goals

python resize_image.py /Users/matthewchang/Desktop/airbnb_images/airbnb2
rsync -vaz /Users/matthewchang/Desktop/airbnb_images/airbnb1_resized/ akshara:/home/akshara/asc_demo/home-robot/projects/spot/airbnb2_goals

python resize_image.py /Users/matthewchang/Desktop/airbnb_images/airbnb3
rsync -vaz /Users/matthewchang/Desktop/airbnb_images/airbnb3_resized/ akshara:/home/akshara/asc_demo/home-robot/projects/spot/airbnb3_goals

DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial0 --goals=object_oven,language_oven1,object_plant,image_chair3,image_refrigerator1,language_refrigerator1,object_toilet,image_couch1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial1 --goals=image_cup1,language_chair1,image_bear1,object_sink,object_toilet,object_bear,language_toilet1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial2 --goals=language_couch1,language_sink2,image_plant2,image_bed1,image_bed2,object_sink,image_toilet1,image_chair3
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial3 --goals=image_couch2,language_plant2,object_bed,image_bear1,object_table,language_bed1,image_chair1,image_plant2,image_oven1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial4 --goals=image_plant2,image_bed1,language_plant1,image_chair1,language_refrigerator1,image_plant1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial5 --goals=image_sink1,image_plant1,object_refrigerator,language_chair2,language_bear1,language_plant1,language_chair1,object_table
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial6 --goals=image_plant2,image_bed2,image_couch2,language_bed2,language_bear1,image_plant1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial7 --goals=image_couch1,object_table,image_sink2,object_sink,language_bed2,object_oven
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial8 --goals=object_plant,language_plant2,image_couch2,language_bed1,image_plant3,language_bear1,image_toilet1,language_couch2,language_chair1
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial9 --goals=language_bed2,image_couch2,image_chair1,image_plant2,language_sink1,language_couch1,image_oven1

DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_trial9_test --goals=language_bed2,image_couch2,image_chair1,image_plant2,language_sink1,language_couch1,image_oven1

DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_video1 --goals=image_bed1,image_bed2,object_oven,image_chair1,language_couch1,image_chair3,language_chair2

DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb1_video2 --goals=image_bed1,image_bed2,object_oven,image_chair1,language_couch1,image_chair3,language_chair2 --offline
k
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb3_test --goals=image_bed1,image_bed2,object_oven,image_chair1,language_couch1,image_chair3,language_chair2
DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb3_test --goals=image_bed1,image_bed2,object_oven,image_chair1,language_couch1,image_chair3,language_chair2 --raw-depth

DISPLAY=:1 python projects/spot/goat.py --trajectory=spot_abnb3_video1 --goals=image_couch3,image_plant1,image_couch1,image_bed1,image_sink2,image_chair2,image_bear1,image_cup1,image_couch2,image_cup2,image_bowl1,image_plant3,image_chair3,image_bowl2,image_sink1,image_tv2,image_chair1

