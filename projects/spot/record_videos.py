import glob
import pickle
import sys
from pathlib import Path

import cv2
import natsort
import numpy as np

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

    main_vis_dir = f"{trajectory}/main_visualization/"
    main_vis = []
    for path in natsort.natsorted(glob.glob(f"{main_vis_dir}/*.png")):
        main_vis.append(cv2.imread(path))

    print(f"Recording videos for {trajectory} with {len(observations)} timesteps")
    breakpoint()

    # Timestamps

    # RGB
    rgbs = [obs.rgb for obs in observations]

    # Semantics
    semantic_frames = [img[50:530, 15:655] for img in main_vis]

    # Depth
    depths = []
    for obs in observations:
        depth_frame = obs.depth
        if depth_frame.max() > 0:
            depth_frame = depth_frame / depth_frame.max()
        depth_frame = (depth_frame * 255).astype(np.uint8)
        depth_frame = np.repeat(depth_frame[:, :, np.newaxis], 3, axis=2)
        depths.append(depth_frame)

    # Goal
    goals = [img[50:530, 670:1310] for img in main_vis]

    # Map
    maps = [img[50:530, 1325:1805] for img in main_vis]

    # Legend
    legend = [img[155 : 155 + 537, 1250 : 1250 + 115] for img in main_vis]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--trajectory", default="trajectories/trajectory1")
    args = parser.parse_args()

    record_videos(args.trajectory)
