import time

import numpy as np

from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchKinematics
from home_robot_hw.stretch_client import StretchClient

if __name__ == "__main__":
    robot = StretchClient()
    robot.manip.home()

    # Async navigation
    robot.switch_to_navigation_mode()

    robot.nav.navigate_to([1, 0, 0], blocking=False)
    time.sleep(1)
    robot.nav.navigate_to([0.2, 0.5, 0], blocking=False)  # update goal
    robot.nav.wait()  # wait for nav actions to complete

    # Async nav and head motion
    robot.nav.navigate_to([0, 0, 0], blocking=False)
    robot.head.look_at_ee()
    robot.wait()  # wait for all actions to complete

    # Async manip and head motion (does not work)
    robot.switch_to_manipulation_mode()

    pos_desired = np.array([0.2, -0.2, 0.4])
    robot.manip.goto_ee_pose(pos_desired, relative=True, blocking=False)
    robot.head.look_ahead()  # stops previous arm motion!
