import rospy
import home_robot
from home_robot.core.interfaces import Action, Observations
from home_robot_hw.env.stretch_abstract_env import StretchEnv

from home_robot.perception.detection.detic.detic_perception import DeticPerception


REAL_WORLD_CATEGORIES=["chair", "mug"]


class StretchObjectNavEnv(StretchEnv):
    """ Create a detic-based object nav environment""" 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(REAL_WORLD_CATEGORIES),
            sem_gpu_id=0,
        )

    def reset(self):
        pass

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        pass

    def get_observation(self) -> Observations:
        pass

    @property
    def episode_over(self) -> bool:
        pass

    def get_episode_metrics(self) -> Dict:
        pass



if __name__ == '__main__':
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchObjectNavEnv(init_cameras=True)
   
