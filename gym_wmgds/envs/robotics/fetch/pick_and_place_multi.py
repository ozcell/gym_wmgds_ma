import os
from gym_wmgds import utils
from gym_wmgds.envs.robotics import fetch_multi_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_multi.xml')


class FetchPickAndPlaceMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15, hide_extra_objs=False):
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
            'object9:joint': [0.90, 0.025, 0.025, 1., 0., 0., 0.]
        }
        global MODEL_XML_PATH
        if hide_extra_objs:
            MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_multi_hidden.xml')
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp)
        utils.EzPickle.__init__(self)


class FetchPickAndPlaceHardMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15, hide_extra_objs=False):
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
            'object9:joint': [0.90, 0.025, 0.025, 1., 0., 0., 0.]
        }
        global MODEL_XML_PATH
        if hide_extra_objs:
            MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_multi_hidden.xml')
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05, 
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp,
            target_in_the_air_percent=1.5, target_in_the_air_lower=0.05)
        utils.EzPickle.__init__(self)


class FetchPickAndPlaceFloorMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
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
            'object9:joint': [0.90, 0.025, 0.025, 1., 0., 0., 0.]
        }
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp)

class FetchPickAndPlaceGrippedMultiEnv(fetch_multi_env.FetchMultiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', n_objects=1, obj_action_type=[0,1,2], observe_obj_grp=False, obj_range=0.15, hide_extra_objs=True):
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
            'object9:joint': [0.90, 0.025, 0.025, 1., 0., 0., 0.]
        }
        global MODEL_XML_PATH
        if hide_extra_objs:
            MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_multi_hidden.xml')
        fetch_multi_env.FetchMultiEnv.__init__(
            self, MODEL_XML_PATH, block_gripper=False, n_substeps=20,
            gripper_extra_height=0., target_in_the_air=True, target_stacked=False, target_offset=0.0,
            obj_range=obj_range, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            n_objects=n_objects, obj_action_type=obj_action_type, observe_obj_grp=observe_obj_grp, gripped_object=True)
        utils.EzPickle.__init__(self)
