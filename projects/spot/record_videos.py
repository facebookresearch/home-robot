import glob
import pickle
import sys
from pathlib import Path

import cv2
import natsort

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


def record_videos(trajectory: str):
    obs_dir = f"{trajectory}/obs/"
    observations = []
    for path in natsort.natsorted(glob.glob(f"{obs_dir}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))
    print(f"Recording videos for {trajectory} with {len(observations)} timesteps")

    # full_vis = natsort.natsorted(glob.glob(f"{trajectory}/map_visualization/*.png"))
    # full_vis = [cv2.imread(f) for f in full_vis]
    # create_video(
    #     full_vis,
    #     f"{trajectory}/full_vis.mp4",
    #     fps=5,
    # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--trajectory", default="trajectories/trajectory1")
    args = parser.parse_args()

    record_videos(args.trajectory)
