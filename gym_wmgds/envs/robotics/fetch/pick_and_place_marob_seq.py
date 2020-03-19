import os
from gym_wmgds import utils
from gym_wmgds.envs.robotics import fetch_marob_env_seq


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_marob_2_seq.xml')

class FetchPickAndPlaceMaRobSeqEnv(fetch_marob_env_seq.FetchMaRobEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15, n_robots=2):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot1:slide0': 0.405,
            'robot1:slide1': 1.28,
            'robot1:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],            
            'object1:joint': [0.10, 0.025, 0.025, 1., 0., 0., 0.]
        }
        fetch_marob_env_seq.FetchMaRobEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp, n_robots=n_robots)
        utils.EzPickle.__init__(self)