import os
import numpy as np

from gym_wmgds import utils, error
from gym_wmgds.envs.robotics import rotations, hand_multi_env
from gym_wmgds.envs.robotics.utils import robot_get_obs, activate_weld_eqs, deactivate_weld_eqs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join('hand', 'manipulate_block_multi.xml')
MANIPULATE_EGG_XML = os.path.join('hand', 'manipulate_egg_multi.xml')
MANIPULATE_PEN_XML = os.path.join('hand', 'manipulate_pen_multi.xml')


class ManipulateMultiEnv(hand_multi_env.HandMultiEnv, utils.EzPickle):
    def __init__(
        self, model_path, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos={},
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
        ignore_z_target_rotation=False, obj_action_type=[0,1,2]
    ):
        """Initializes a new Hand manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
        """
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.ignore_z_target_rotation = ignore_z_target_rotation

        self.n_objects = 1
        self.ai_object = False
        self.obj_action_type = obj_action_type

        self.n_actions = self.n_objects * len(obj_action_type) + 20

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']

        hand_multi_env.HandMultiEnv.__init__(
            self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control, n_actions=self.n_actions)
        utils.EzPickle.__init__(self)

    def _get_achieved_goal(self):
        # Object position and rotation.
        if self.ai_object:
            object_qpos = self.sim.data.get_joint_qpos('iobject:joint')
        else:
            object_qpos = self.sim.data.get_joint_qpos('object:joint')
        assert object_qpos.shape == (7,)
        return object_qpos

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return (success - 1.)
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10. * d_pos + d_rot)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # swap object with intelligent object
        if self.ai_object:
            iobject_initial_qpos = self.sim.data.get_joint_qpos('iobject:joint').copy()
            object_initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
            self.sim.data.set_joint_qpos('iobject:joint', object_initial_qpos)
            self.sim.data.set_joint_qpos('object:joint', iobject_initial_qpos)
            ## Added to disable welding constrain before resetting the environment
            #self.sim.model.eq_active[0] = 0
        self.sim.forward()

        if self.ai_object:
            initial_qpos = self.sim.data.get_joint_qpos('iobject:joint').copy()
        else:
            initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        if self.ai_object:
            self.sim.data.set_joint_qpos('iobject:joint', initial_qpos)
        else:
            self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        def is_on_palm():
            self.sim.forward()
            if self.ai_object:
                cube_middle_idx = self.sim.model.site_name2id('iobject:center')
            else:
                cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # First disable the weld equations for ai objects so they can place inside the env realistically 
        if self.ai_object:
            deactivate_weld_eqs(self.sim)
        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(self.n_actions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        # Now enable the weld equations for ai objects run the simulation for a bunch of timesteps to let everything settle in
        if self.ai_object:
            activate_weld_eqs(self.sim)
            for _ in range(10):
                self._set_action(np.zeros(self.n_actions))
                try:
                    self.sim.step()
                except mujoco_py.MujocoException:
                    return False
        ## Added to enable welding constrain after resetting the environment
        #self.sim.model.eq_active[0] = 1
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            if self.ai_object:
                target_pos = self.sim.data.get_joint_qpos('iobject:joint')[:3] + offset
            else:
                target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            if self.ai_object:
                target_pos = self.sim.data.get_joint_qpos('iobject:joint')[:3]
            else:
                target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            if self.ai_object:
                target_quat = self.sim.data.get_joint_qpos('iobject:joint')
            else:
                target_quat = self.sim.data.get_joint_qpos('object:joint')
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if self.ai_object:
            if 'iobject_hidden' in self.sim.model.geom_names:
                hidden_id = self.sim.model.geom_name2id('iobject_hidden')
                self.sim.model.geom_rgba[hidden_id, 3] = 1.
        else:
            if 'object_hidden' in self.sim.model.geom_names:
                hidden_id = self.sim.model.geom_name2id('object_hidden')
                self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.ai_object:
            object_qvel = self.sim.data.get_joint_qvel('iobject:joint')
        else:
            object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        obs_all = []
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        obs_all.append(observation.copy())
        obj_observation = np.concatenate([np.zeros_like(robot_qpos), np.zeros_like(robot_qvel), object_qvel, achieved_goal])
        obs_all.append(obj_observation.copy())
        obs_all = np.asarray(obs_all)

        return {
            'observation': obs_all.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }


class HandBlockMultiEnv(ManipulateMultiEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse', obj_action_type=[3,4,5,6], observe_obj_grp=True):
        super(HandBlockMultiEnv, self).__init__(
            model_path=MANIPULATE_BLOCK_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            obj_action_type=obj_action_type)


class HandEggMultiEnv(ManipulateMultiEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse', obj_action_type=[3,4,5,6], observe_obj_grp=True):
        super(HandEggMultiEnv, self).__init__(
            model_path=MANIPULATE_EGG_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            obj_action_type=obj_action_type)


class HandPenMultiEnv(ManipulateMultiEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse', obj_action_type=[3,4,5,6], observe_obj_grp=True):
        super(HandPenMultiEnv, self).__init__(
            model_path=MANIPULATE_PEN_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False, reward_type=reward_type,
            ignore_z_target_rotation=True, distance_threshold=0.05,
            obj_action_type=obj_action_type)
