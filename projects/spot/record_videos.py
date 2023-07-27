import cv2
import glob
import natsort


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()

trajectories_folder = "07_26_trajectories"

for traj in glob.glob(f"{trajectories_folder}/*"):
    print(f"Recording videos for {traj}")
    full_vis = natsort.natsorted(glob.glob(f"{traj}/map_visualization/*.png"))
    full_vis = [cv2.imread(f) for f in full_vis]
    create_video(
        full_vis,
        f"{traj}/full_vis.mp4",
        fps=5,
    )

    planner_vis = natsort.natsorted(glob.glob(f"{traj}/images/planner/*.png"))
    planner_vis = [cv2.imread(f) for f in planner_vis]
    create_video(
        planner_vis,
        f"{traj}/planner_vis.mp4",
        fps=5,
    )
