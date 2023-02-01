import home_robot

class Env:
    def __init__(self):
        pass

    def apply_action(self, action: home_robot.Action):
        pass

    def get_observation(self) -> home_robot.Observation:
        pass