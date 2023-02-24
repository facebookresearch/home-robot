from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchManipulationEnv(StretchEnv):
    """Simple manipulation environment"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
