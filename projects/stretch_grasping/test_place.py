import click
import numpy as np

from home_robot.core.interfaces import Observations
from home_robot.motion.stretch import STRETCH_PREGRASP_Q
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.remote import StretchClient
from home_robot_hw.utils.placing import PlacePlanner

GOAL_OPTIONS = [
    "other",
    "chair",
    "cup",
    "table",
    "other",
]


@click.command()
@click.option("--goal-recep", default="chair")
def main(goal_recep):
    assert goal_recep in GOAL_OPTIONS

    # Set up robot
    robot = StretchClient()

    robot.switch_to_manipulation_mode()
    robot.manip.goto_joint_positions(robot.manip._extract_joint_pos(STRETCH_PREGRASP_Q))

    # Prompt for robot to grasp object
    print("Gripper opening...")
    robot.manip.open_gripper()
    input("Place object in gripper then press enter.")
    print("Gripper closing...")
    robot.manip.close_gripper()

    # Set up place modules
    place_planner = PlacePlanner(robot)
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(GOAL_OPTIONS),
        sem_gpu_id=0,
    )

    # Get goal recepticle mask
    robot.head.look_at_ee()
    rgb, dpt, xyz = robot.head.get_images(compute_xyz=True)
    camera_pose = robot.head.get_pose()
    print("camera pose: ", camera_pose)

    obs = Observations(
        rgb=rgb.copy(),
        depth=dpt.copy(),
        xyz=xyz.copy(),
        gps=np.zeros(2),
        compass=np.zeros(1),
        task_observations={},
        camera_pose=camera_pose,
    )

    obs = segmentation.predict(obs)
    object_mask = obs.semantic == GOAL_OPTIONS.index(goal_recep)

    obj_pc = xyz.reshape(-1, 3) * object_mask.reshape(-1, 1)
    obj_pc_rgb = rgb.reshape(-1, 3) * object_mask.reshape(-1, 1)

    # Execute place
    place_planner.try_placing(obj_pc, obj_pc_rgb)


if __name__ == "__main__":
    main()
