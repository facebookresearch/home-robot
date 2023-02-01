import habitat

import home_robot

class HabitatAgent(habitat.Agent):
    def __init__(self, home_robot_agent: home_robot.agent.Agent):
        self.hr_agent = home_robot_agent

    def reset(self):
        pass

    def act(self):
        pass