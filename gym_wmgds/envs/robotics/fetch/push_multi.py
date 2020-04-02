import os
from gym_wmgds import utils
from gym_wmgds.envs.robotics import fetch_multi_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_multi.xml')
MODEL_OBSTACLE_XML_PATH = os.path.join('fetch', 'push_obstacle_multi.xml')


class FetchPushMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15):
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
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, 
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp)
        utils.EzPickle.__init__(self)


class FetchPushObstacleMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15):
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
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_OBSTACLE_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp,
            obj_range_lower=[-0.15,-0.15], obj_range_upper=[0.15,0.05],
            target_range_lower=[-0.15,0.10,0.00], target_range_upper=[0.15,0.25,0.00],
            )        
            utils.EzPickle.__init__(self)
