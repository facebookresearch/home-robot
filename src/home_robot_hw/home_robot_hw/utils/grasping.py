import timeit
from typing import Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped

import home_robot.utils.visualization as viz
from home_robot.motion.stretch import (
    STRETCH_NAVIGATION_Q,
    STRETCH_PREGRASP_Q,
    HelloStretchIdx,
    HelloStretchKinematics,
)
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.utils import matrix_to_pose_msg, ros_pose_to_transform

STRETCH_GRIPPER_LENGTH = 0.2


class GraspPlanner(object):
    """Simple grasp planner which integrates with a ROS service runnning e.g. contactgraspnet.
    Will choose and execute a grasp based on distance from base."""

    def __init__(
        self,
        robot_client,
        env,
        visualize_planner=False,
        debug_point_cloud=False,
        verbose=True,
    ):
        self.robot_client = robot_client
        self.env = env
        self.robot_model = HelloStretchKinematics(visualize=visualize_planner)
        self.grasp_client = RosGraspClient()
        self.verbose = verbose

        # Add this flag to make sure that the point clouds are coming in correctly - will visualize what the points look like relative to a base coordinate frame with z = up, x = forward
        self.debug_point_cloud = debug_point_cloud

    def go_to_manip_mode(self):
        """Move the arm and head into manip mode."""
        """
        home_q = STRETCH_PREGRASP_Q
        home_q = self.robot_model.update_look_at_ee(home_q)
        self.robot_client.goto(home_q, move_base=False, wait=True)
        """
        self.robot_client.switch_to_manipulation_mode()
        self.robot_client.head.look_at_ee(blocking=False)
        self.robot_client.manip.goto_joint_positions(
            self.robot_client.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
        )
        self.robot_client.switch_to_navigation_mode()

    def go_to_nav_mode(self):
        """Move the arm and head into nav mode."""
        """
        home_q = STRETCH_NAVIGATION_Q
        # TODO - should be looking down to make sure we can see the objects
        home_q = self.robot_model.update_look_front(home_q.copy())
        # NOTE: for now we have to do this though - until bugs are fixed in semantic map
        # home_q = self.robot_model.update_look_ahead(home_q.copy())
        self.robot_client.goto(home_q, move_base=False, wait=True)
        """
        self.robot_client.switch_to_manipulation_mode()
        self.robot_client.head.look_front(blocking=False)
        self.robot_client.manip.goto_joint_positions(
            self.robot_client.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
        )
        self.robot_client.switch_to_navigation_mode()

    def try_grasping(
        self, visualize: bool = False, dry_run: bool = False, max_tries: int = 10
    ):
        """Detect grasps and try to pick up an object in front of the robot.
        Visualize - will show debug point clouds
        Dry run - does not actually move, just computes everything"""
        """
        home_q = STRETCH_PREGRASP_Q
        home_q = self.robot_model.update_look_front(home_q.copy())
        home_q = self.robot_model.update_gripper(home_q, open=True)
        self.robot_client.goto(home_q, move_base=False, wait=True)
        home_q = self.robot_model.update_look_at_ee(home_q)
        """
        if not self.robot_client.in_manipulation_mode():
            self.robot_client.switch_to_manipulation_mode()
        self.robot_client.head.look_at_ee(blocking=False)
        self.robot_client.manip.open_gripper()
        visualize = True

        min_grasp_score = 0.0
        min_obj_pts = 100
        for attempt in range(max_tries):
            self.robot_client.head.look_at_ee(blocking=False)
            self.robot_client.manip.goto_joint_positions(
                self.robot_client.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
            )
            rospy.sleep(1.0)

            # Get the observations - we need depth and xyz point clouds
            t0 = timeit.default_timer()
            obs = self.env.get_observation()
            rgb, depth, xyz = obs.rgb, obs.depth, obs.xyz

            # TODO: verify this is correct
            # In world coordinates
            # camera_pose_world = self.robot_client.head.get_pose()
            # camera_pose = camera_pose_world
            # In base coordinates
            camera_pose_base = self.robot_client.head.get_pose_in_base_coords()
            camera_pose = camera_pose_base

            # TODO: remove debug code
            if self.debug_point_cloud:
                import trimesh

                from home_robot.utils.point_cloud import show_point_cloud

                show_point_cloud(xyz, rgb / 255.0, orig=np.zeros(3))
                xyz2 = trimesh.transform_points(xyz.reshape(-1, 3), camera_pose)
                show_point_cloud(xyz2, rgb / 255.0, orig=np.zeros(3))
                camera_pose_world = self.robot_client.head.get_pose()
                xyz3 = trimesh.transform_points(xyz.reshape(-1, 3), camera_pose_world)
                show_point_cloud(xyz3, rgb / 255.0, orig=np.zeros(3))
                breakpoint()

            if self.verbose:
                print(
                    "Getting images + cam pose took",
                    timeit.default_timer() - t0,
                    "seconds",
                )

            object_mask = obs.task_observations["goal_mask"]

            if visualize:
                viz.show_image_with_mask(rgb, object_mask)

            num_object_pts = np.sum(object_mask)
            print("found this many object points:", num_object_pts)
            if num_object_pts < min_obj_pts:
                continue

            mask_valid = (
                depth > self.robot_client._ros_client.dpt_cam.near_val
            )  # remove bad points
            mask_scene = mask_valid  # initial mask has to be good
            mask_scene = mask_scene.reshape(-1)

            predicted_grasps, in_base_frame = self.grasp_client.request(
                xyz,
                rgb,
                object_mask,
                frame=self.robot_client._ros_client.rgb_cam.get_frame(),
                camera_pose=camera_pose,
            )
            if 0 not in predicted_grasps:
                print("no predicted grasps")
                continue
            predicted_grasps, scores = predicted_grasps[0]
            print("got this many grasps:", len(predicted_grasps))

            grasps = []
            for i, (score, grasp) in sorted(
                enumerate(zip(scores, predicted_grasps)), key=lambda x: x[0]
            ):
                pose = grasp
                if not in_base_frame:
                    pose = camera_pose @ pose
                if score < min_grasp_score:
                    continue

                # Get angles in world frame
                theta_x, theta_y = divergence_from_vertical_grasp(pose)
                theta = max(theta_x, theta_y)
                print(i, "score =", score, theta, "xy =", theta_x, theta_y)
                self._send_predicted_grasp_to_tf(pose)
                # Reject grasps that arent top down for now
                if theta > 0.3:
                    continue
                grasps.append(pose)

            print("After filtering: # grasps =", len(grasps))

            # Poses from the grasp server are pinch points
            # retract by gripper length to get gripper pose
            grasp_offset = np.eye(4)
            grasp_offset[2, 3] = -STRETCH_GRIPPER_LENGTH

            for i, grasp in enumerate(grasps):
                grasps[i] = grasp @ grasp_offset

            for grasp in grasps:
                print("Executing grasp:")
                print(grasp)
                theta_x, theta_y = divergence_from_vertical_grasp(grasp)
                print(" - with theta x/y from vertical =", theta_x, theta_y)
                if not dry_run:
                    grasp_completed = self.try_executing_grasp(grasp)
                else:
                    grasp_completed = False
                if grasp_completed:
                    break
            break

        self.robot_client.switch_to_navigation_mode()

    def plan_to_grasp(self, grasp: np.ndarray) -> Optional[np.ndarray]:
        """Create offsets for the full trajectory plan to get to the object.
        Then return that plan."""
        grasp_pos, grasp_quat = to_pos_quat(grasp)
        self.robot_client.switch_to_manipulation_mode()
        self.robot_client.manip.open_gripper()

        # Get pregrasp pose: current pose + maxed out lift
        pos_pre, quat_pre = self.robot_client.manip.get_ee_pose()
        joint_pos_pre = self.robot_client.manip.get_joint_positions()
        # Save initial waypoint to return to
        initial_pt = ("initial", joint_pos_pre, False)

        # Create a pregrasp point at the top of the robot's arc
        pregrasp_cfg = joint_pos_pre.copy()
        pregrasp_cfg[1] = 0.95
        pregrasp = ("pregrasp", pregrasp_cfg, False)

        # Try grasp first - find an IK solution for this
        grasp_cfg = self.robot_client.manip.solve_ik(grasp_pos, grasp_quat)
        if grasp_cfg is not None:
            grasp_pt = (
                "grasp",
                self.robot_client.manip._extract_joint_pos(grasp_cfg),
                True,
            )
        else:
            print("-> could not solve for grasp")
            return None

        # Standoff is 8cm over the grasp for now
        standoff_pos = grasp_pos + np.array([0.0, 0.0, 0.08])
        standoff_cfg = self.robot_client.manip.solve_ik(
            standoff_pos,
            grasp_quat,  # initial_cfg=grasp_cfg
        )
        if standoff_cfg is not None:
            standoff = (
                "standoff",
                self.robot_client.manip._extract_joint_pos(standoff_cfg),
                False,
            )
        else:
            print("-> could not solve for standoff")
            return None
        back_cfg = self.robot_client.manip._extract_joint_pos(standoff_cfg)
        back_cfg[2] = 0.01
        back = ("back", back_cfg, False)

        return [pregrasp, back, standoff, grasp_pt, standoff, back, initial_pt]

    def _send_predicted_grasp_to_tf(self, grasp):
        """Helper function for visualizing the predicted grasps."""
        # Convert grasp pose to pos/quaternion
        # Visualize the grasp in RViz
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "predicted_grasp"
        t.header.frame_id = "base_link"
        t.transform = ros_pose_to_transform(matrix_to_pose_msg(grasp))
        self.grasp_client.broadcaster.sendTransform(t)

    def _publish_current_ee_pose(self):
        """Helper function for debugging EE pose issues on the robot"""
        pos, quat = self.robot_client.manip.get_ee_pose(world_frame=False)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "current_ee_pose"
        t.header.frame_id = "base_link"
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.grasp_client.broadcaster.sendTransform(t)

    def try_executing_grasp(
        self, grasp: np.ndarray, wait_for_input: bool = False
    ) -> bool:
        """Execute a predefined grasp trajectory to the given pose. Grasp should be an se(3) pose, expressed as a 4x4 numpy matrix."""
        assert grasp.shape == (4, 4)
        self._send_predicted_grasp_to_tf(grasp)
        wait_for_input = True

        # Generate a plan
        trajectory = self.plan_to_grasp(grasp)

        if trajectory is None:
            print("Planning failed")
            return False

        for i, (name, waypoint, should_grasp) in enumerate(trajectory):
            self.robot_client.manip.goto_joint_positions(waypoint)
            # TODO: remove this delay - it's to make sure we don't start moving again too early
            rospy.sleep(0.1)
            self._publish_current_ee_pose()
            if should_grasp:
                self.robot_client.manip.close_gripper()
            if wait_for_input:
                input(f"{i+1}) went to {name}")
            else:
                print(f"{i+1}) went to {name}")
        print("!!! GRASP ATTEMPT COMPLETE !!!")
        return True


def divergence_from_vertical_grasp(grasp: np.ndarray) -> Tuple[float, float]:
    """Grasp should be a matrix in SE(3). Compute if its roughly vertical. Returns angles from vertical."""
    dirn = grasp[:3, 2]
    theta_x = np.abs(np.arctan(dirn[0] / dirn[2]))
    theta_y = np.abs(np.arctan(dirn[1] / dirn[2]))
    return theta_x, theta_y
