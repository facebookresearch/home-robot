import glob
import os
import shutil

import cv2
from natsort import natsorted


def record_video(source_dir: str, target_dir: str, target_file: str):
    # shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Recording video {target_dir}/{target_file}")

    # Semantic map vis
    image_paths = natsorted(glob.glob(f"{source_dir}/tp_snapshot*.png"))
    if len(image_paths) == 0:
        image_paths = natsorted(glob.glob(f"{source_dir}/snapshot*.png"))
    if len(image_paths) == 0:
        return
    # Get the dimensions of the first image (assuming all images have the same dimensions)
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape
    size = (width, height)

    out = cv2.VideoWriter(
        f"{target_dir}/{target_file}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        size,
    )
    for filename in image_paths:
        frame = cv2.imread(filename)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    record_video(
        source_dir="datadump/images/eval_hssd/107733960_175999701_3",
        target_dir="video_dir",
        target_file="test",
    )
