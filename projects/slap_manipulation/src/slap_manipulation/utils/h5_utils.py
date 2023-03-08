import h5py
from matplotlib import pyplot as plt

from home_robot.utils.data_tools.image import img_from_bytes


def view_keyframe_imgs(file_object: h5py.File, trial_name: str):
    num_keyframes = len(file_object[f"{trial_name}/rgb"].keys())
    for i in range(num_keyframes):
        _key = f"{trial_name}/rgb/{i}"
        img = img_from_bytes(file_object[_key][()])
        plt.imshow(img)
        plt.show()
