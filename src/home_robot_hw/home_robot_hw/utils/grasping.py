import timeit

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped

import home_robot.utils.visualization as viz
from home_robot.motion.stretch import (
    STRETCH_NAVIGATION_Q,
    STRETCH_PREGRASP_Q,
    HelloStretch,
    HelloStretchIdx,
)
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.utils import matrix_to_pose_msg, ros_pose_to_transform


class GraspPlanner(object):
    """Simple grasp planner which integrates with a ROS service runnning e.g. contactgraspnet.
    Will choose and execute a grasp based on distance from base."""

    def __init__(self, robot_client, visualize_planner=False):
        self.robot_client = robot_client
        self.robot_model = HelloStretch(visualize=visualize_planner)
        self.grasp_client = RosGraspClient()

    def go_to_manip_mode(self):
        """Move the arm and head into manip mode."""
        home_q = STRETCH_PREGRASP_Q
        home_q = self.robot_model.update_look_at_ee(home_q)
        self.robot_client.goto(home_q, move_base=False, wait=True)

    def go_to_nav_mode(self):
        """Move the arm and head into nav mode."""
        home_q = STRETCH_NAVIGATION_Q
        # TODO - should be looking down to make sure we can see the objects
        # home_q = self.robot_model.update_look_front(home_q.copy())
        # NOTE: for now we have to do this though - until bugs are fixed in semantic map
        home_q = self.robot_model.update_look_ahead(home_q.copy())
        self.robot_client.goto(home_q, move_base=False, wait=True)

    def try_grasping(self, visualize=False, dry_run=False):
        """Detect grasps and try to pick up an object in front of the robot."""
        home_q = STRETCH_PREGRASP_Q
        home_q = self.robot_model.update_look_front(home_q.copy())
        home_q = self.robot_model.update_gripper(home_q, open=True)
        self.robot_client.goto(home_q, move_base=False, wait=True)
        home_q = self.robot_model.update_look_at_ee(home_q)

        min_grasp_score = 0.0
        max_tries = 10
        min_obj_pts = 100
        for attempt in range(max_tries):
            print("look at ee")
            self.robot_client.goto(home_q, move_base=False, wait=True)
            rospy.sleep(1.0)

            # Get the observations - we need depth and xyz point clouds
            t0 = timeit.default_timer()
            obs = self.robot_client.get_observation()
            rgb, depth, xyz = obs.rgb, obs.depth, obs.xyz
            camera_pose = self.robot_client.get_pose(
                self.robot_client.rgb_cam.get_frame()
            )
            print(
                "getting images + cam pose took", timeit.default_timer() - t0, "seconds"
            )

            object_mask = obs.task_observations["goal_mask"]

            if visualize:
                viz.show_image_with_mask(rgb, object_mask)

            num_object_pts = np.sum(object_mask)
            print("found this many object points:", num_object_pts)
            if num_object_pts < min_obj_pts:
                continue

            mask_valid = depth > self.robot_client.dpt_cam.near_val  # remove bad points
            mask_scene = mask_valid  # initial mask has to be good
            mask_scene = mask_scene.reshape(-1)

            predicted_grasps = self.grasp_client.request(
                xyz, rgb, object_mask, frame=self.robot_client.rgb_cam.get_frame()
            )
            if 0 not in predicted_grasps:
                print("no predicted grasps")
                continue
            predicted_grasps, scores = predicted_grasps[0]
            print("got this many grasps:", len(predicted_grasps))

            grasps = []
            for i, (score, grasp) in enumerate(zip(scores, predicted_grasps)):
                pose = grasp
                pose = camera_pose @ pose
                if score < min_grasp_score:
                    continue

                # Get angles in world frame
                theta_x, theta_y = divergence_from_vertical_grasp(pose)
                theta = max(theta_x, theta_y)
                print(i, "score =", score, theta, "xy =", theta_x, theta_y)
                # Reject grasps that arent top down for now
                if theta > 0.3:
                    continue
                grasps.append(pose)

            # Correct for the length of the Stretch gripper and the gripper upon
            # which Graspnet was trained
            grasp_offset = np.eye(4)
            grasp_offset[2, 3] = -0.09
            for i, grasp in enumerate(grasps):
                grasps[i] = grasp @ grasp_offset

            for grasp in grasps:
                print("Executing grasp")
                print(grasp)
                theta_x, theta_y = divergence_from_vertical_grasp(grasp)
                print("with xy =", theta_x, theta_y)
                if not dry_run:
                    grasp_completed = self.try_executing_grasp(grasp)
                else:
                    grasp_completed = False
                if grasp_completed:
                    break

    def try_executing_grasp(self, grasp) -> bool:
        """Try executing a grasp. Takes in robot self.robot_model and a potential grasp; will execute
        this grasp if possible. Grasp-client is just used to send a debugging TF frame for now.

        Returns true if the grasp was possible and was executed; false if not."""
        # Get our current joint states
        q, _ = self.robot_client.update()

        # Convert grasp pose to pos/quaternion
        grasp_pose = to_pos_quat(grasp)
        print("grasp xyz =", grasp_pose[0])

        # If can't plan to reach grasp, return
        qi = self.robot_model.manip_ik(grasp_pose, q)
        qi = self.robot_model.update_gripper(qi, open=True)
        print("x motion =", qi[0])
        if qi is not None:
            self.robot_model.set_config(qi)
        else:
            print(" --> ik failed")
            return False
        # TODO: remove this when the base is moving again!!!
        if np.abs(qi[0]) > 0.0025:
            return False
        input("press enter to move")

        # Standoff 8 cm above grasp position
        q_standoff = qi.copy()
        # q_standoff[HelloStretchIdx.LIFT] += 0.08   # was 8cm, now more
        q_standoff[HelloStretchIdx.LIFT] += 0.1

        # Actual grasp position
        q_grasp = qi.copy()

        if q_standoff is not None:
            # If standoff position invalid, return
            if not self.robot_model.validate(q_standoff):
                print("invalid standoff config:", q_standoff)
                return False
            print("found standoff")

            # Visualize the grasp in RViz
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "predicted_grasp"
            t.header.frame_id = "map"
            t.transform = ros_pose_to_transform(matrix_to_pose_msg(grasp))
            self.grasp_client.broadcaster.sendTransform(t)

            # Go to the grasp and try it
            # First we need to create and move to a decent pre-grasp pose
            q[HelloStretchIdx.LIFT] = 0.99
            self.robot_client.goto(
                q, move_base=False, wait=True, verbose=False
            )  # move arm to top
            # input('--> go high')
            q_pre = q.copy()
            # NOTE: this gets the gripper in the right orientation before we start to move - nothing
            # else should change except lift and arm!
            q_pre[5:] = q_standoff[5:]  # TODO: Add constants for joint indices
            q_pre = self.robot_model.update_gripper(q_pre, open=True)
            # self.robot_client.move_base(theta=q_standoff[2])  # TODO Replace this
            rospy.sleep(2.0)
            self.robot_client.goto(q_pre, move_base=False, wait=False, verbose=False)

            # Move to standoff pose
            self.robot_model.set_config(q_standoff)
            print("Q =", q_standoff)
            print(q_grasp != q_standoff)
            input("--> gripper ready; go to standoff")
            q_standoff = self.robot_model.update_gripper(q_standoff, open=True)
            self.robot_client.goto(
                q_standoff, move_base=False, wait=True, verbose=False
            )

            # move down to grasp pose
            self.robot_model.set_config(q_grasp)
            print("Q =", q_grasp)
            print(q_grasp != q_standoff)
            input("--> go to grasp")
            self.robot_client.goto(q_grasp, move_base=False, wait=True, verbose=True)

            # move down to close gripper
            q_grasp_closed = self.robot_model.update_gripper(q_grasp, open=False)
            print("Q =", q_grasp_closed)
            print(q_grasp != q_grasp_closed)
            input("--> close the gripper")
            self.robot_client.goto(
                q_grasp_closed, move_base=False, wait=False, verbose=True
            )
            rospy.sleep(2.0)

            # Move back to standoff pose
            q_standoff = self.robot_model.update_gripper(q_standoff, open=False)
            self.robot_client.goto(
                q_standoff, move_base=False, wait=True, verbose=False
            )

            # Move back to original pose
            q_pre = self.robot_model.update_gripper(q_pre, open=False)
            self.robot_client.goto(q_pre, move_base=False, wait=True, verbose=False)

            # We completed the grasp
            return True


def divergence_from_vertical_grasp(grasp):
    dirn = grasp[:3, 2]
    theta_x = np.abs(np.arctan(dirn[0] / dirn[2]))
    theta_y = np.abs(np.arctan(dirn[1] / dirn[2]))
    return theta_x, theta_y
