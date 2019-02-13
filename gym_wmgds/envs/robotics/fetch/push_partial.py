import os
from gym_wmgds import utils
from gym_wmgds.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_partial.xml')


class FetchPushPartialEnv(fetch_env.FetchPartialEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', max_nb_objects=10):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [0.10, 0.025, 0.025, 1., 0., 0., 0.],
            'object2:joint': [0.20, 0.025, 0.025, 1., 0., 0., 0.],
            'object3:joint': [0.30, 0.025, 0.025, 1., 0., 0., 0.],
            'object4:joint': [0.40, 0.025, 0.025, 1., 0., 0., 0.],
            'object5:joint': [0.50, 0.025, 0.025, 1., 0., 0., 0.],
            'object6:joint': [0.60, 0.025, 0.025, 1., 0., 0., 0.],
            'object7:joint': [0.70, 0.025, 0.025, 1., 0., 0., 0.],
            'object8:joint': [0.80, 0.025, 0.025, 1., 0., 0., 0.],
            'object9:joint': [0.90, 0.025, 0.025, 1., 0., 0., 0.],
        }
        fetch_env.FetchPartialEnv.__init__(
            self, MODEL_XML_PATH, has_object=2, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
