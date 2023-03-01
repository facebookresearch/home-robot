import hydra
from slap_manipulation.env.stretch_manipulation import StretchManipulationEnv
from slap_manipulation.policy.interaction_prediction_module import IPModule
from slap_manipulation.policy.action_prediction_module import APModule


@hydra.main(version_base=None, config_path="./conf", config_name="all_tasks_01_31")
def main(cfg):
    # create the robot object
    robot = StretchManipulationEnv(cfg)
    # create IPM object
    ipm_model = IPModule()
    # create APM object
    apm_model = APModule()
    # load model-weights
    ipm_model.load_state_dict(cfg.ipm_weights)
    apm_model.load_state_dict(cfg.apm_weights)

    print("Loaded models successfully")
    cmds = [
        "pick up the bottle",
        "open top drawer",
        "open bottom drawer",
        "close the drawers",
        "place in the drawer",
        "pick up lemon from basket",
        "place lemon in bowl",
        "place in basket",
    ]
    experiment_running = True
    while experiment_running:
        for i, cmd in enumerate(cmds):
            print(f"{i+1}. {cmd}")
        task_id = int(input("which task to solve, enter integer: "))
        input_cmd = [cmds[task_id - 1]]
        print(f"Executing {input_cmd}")
        # get from the robot: pcd=(xyz, rgb), gripper-state,
        # construct input vector from raw data
        raw_observations = robot.get_observations()
        input_vector = create_ipm_input(raw_observations)
        # run inference on sensor data for IPM
        interaction_point = ipm_model.eval(input_vector)
        # ask if ok to run APM inference
        for i in range(cfg.num_keypoints):
            # run APM inference on sensor
            raw_observations = robot.get_observations()
            input_vector = create_apm_input(raw_observations, interaction_point)
            action = apm_model.eval(input_vector)
            # ask if ok to execute
            res = input("Execute the output? (y/n)")
            if res == 'y':
                robot.apply_action(action)
            pass
