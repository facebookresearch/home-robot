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


def create_video(images, output_file, fps=5):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


def create_video_aligned(images, output_file, timestamps, fps=30, speedup_factor=1.0):
    assert len(images) == len(
        timestamps
    ), "Mismatch between number of images and timestamps."

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Calculate differences between consecutive timestamps to determine the frame delay.
    frame_delays = [
        timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
    ]
    # Convert time differences to number of frames (assuming 1 unit of timestamp is 1 second for now).
    frame_counts = [round(delay * fps / speedup_factor) for delay in frame_delays]

    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for idx, image in enumerate(images[:-1]):
        for _ in range(frame_counts[idx]):
            video_writer.write(image)
    video_writer.write(images[-1])  # Write the last frame
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

    # Timestamps
    # timestamps = [obs.task_observations["timestamp"] for obs in observations]
    timestamps = []
    for path in natsort.natsorted(glob.glob(f"{obs_dir}/*.pkl")):
        timestamps.append(Path(path).stat().st_mtime)

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
    legends = [img[530 : 530 + 122, 155 : 155 + 1250] for img in main_vis]

    # Videos
    video_dir = f"{trajectory}/videos"
    speedup_factor = 8.0
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    create_video_aligned(
        rgbs, f"{video_dir}/rgb.mp4", timestamps, speedup_factor=speedup_factor
    )
    create_video_aligned(
        semantic_frames,
        f"{video_dir}/semantic_frame.mp4",
        timestamps,
        speedup_factor=speedup_factor,
    )
    create_video_aligned(
        depths, f"{video_dir}/depth.mp4", timestamps, speedup_factor=speedup_factor
    )
    create_video_aligned(
        goals, f"{video_dir}/goal.mp4", timestamps, speedup_factor=speedup_factor
    )
    create_video_aligned(
        maps, f"{video_dir}/map.mp4", timestamps, speedup_factor=speedup_factor
    )
    create_video_aligned(
        legends, f"{video_dir}/legend.mp4", timestamps, speedup_factor=speedup_factor
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--trajectory", default="trajectories/trajectory1")
    args = parser.parse_args()

    record_videos(args.trajectory)
