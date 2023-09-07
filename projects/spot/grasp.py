from home_robot.utils.config import get_config
from home_robot_hw.env.spot_goat_env import SpotGoatEnv
from spot_wrapper.spot import Spot, SpotCamIds
from home_robot.agent.goat_agent.goat_agent import GoatAgent
import numpy as np
import time
config_path = "projects/spot/configs/config.yaml"
config, config_str = get_config(config_path)
config.defrost()
spot = Spot("RealNavEnv")
INITIAL_ARM_JOINT_ANGLES= [0, -170, 120, 0, 75, 0]
with spot.get_lease(hijack=True):
    # env = SpotGoatEnv(spot, position_control=True)
    print("Resetting environment...")
    # image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
    # hand_image_response = image_responses[0]  # only expecting one image
    # env.reset()
    spot.power_on()
    spot.blocking_stand()
    cmd_id = spot.move_gripper_to_point(point=np.array([0.6,0.0,0.7]), rotation=[0.0, 0.0, 0.0])
    spot.block_until_arm_arrives(cmd_id, timeout_sec=10)
    spot.open_gripper()
    time.sleep(1)
    spot.close_gripper()
    spot.block_until_arm_arrives(cmd_id, timeout_sec=10)
    cmd_id = spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES, travel_time=3
        )
    spot.block_until_arm_arrives(cmd_id, timeout_sec=10)
    spot.sit()
    spot.power_off()
    #spot.grasp_point_in_image(hand_image_response, pixel_xy=(50,50), timeout=100)
