import numpy as np

from gym_wmgds import error
from gym_wmgds.envs.robotics import rotations, robot_env, utils
import pdb

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def to_radian(degree):
    return (degree*np.pi)/180.

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class KortexMultiEnv(robot_env.RobotEnv):
    """Superclass for all Kortex environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        target_in_the_air, target_stacked, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, n_objects, obj_action_type, observe_obj_grp
    ):
        """Initializes a new Kortex environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_stacked = target_stacked
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        
        self.n_objects = n_objects
        self.ai_object = False
        self.obj_action_type = obj_action_type
        self.max_n_objects = 5
        self.observe_obj_grp = observe_obj_grp

        self.n_actions = n_objects * len(obj_action_type) + 8

        self.initial_qpos = initial_qpos

        super(KortexMultiEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:gripper_right_driver_joint', to_radian(45))
            self.sim.data.set_joint_qpos('robot0:gripper_right_follower_joint', to_radian(-45))
            self.sim.data.set_joint_qpos('robot0:gripper_right_spring_joint', to_radian(45))
            self.sim.data.set_joint_qpos('robot0:gripper_left_driver_joint', to_radian(45))
            self.sim.data.set_joint_qpos('robot0:gripper_left_follower_joint', to_radian(-45))
            self.sim.data.set_joint_qpos('robot0:gripper_left_spring_joint', to_radian(45)) 
            self.sim.forward()

    def _set_action(self, action):

        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        #pdb.set_trace()

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        #actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        actuation_center = np.zeros_like(action[:9])
        for i in range(self.sim.data.ctrl.shape[0]):
            actuation_center[i] = self.sim.data.get_joint_qpos(
                self.sim.model.actuator_names[i].replace(':A_', ':'))
        
        pos_ctrl, gripper_ctrl = action[:7], action[7]
        #pos_ctrl *= 0.05 

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        self.sim.data.ctrl[:7] = actuation_center[:7] + pos_ctrl * actuation_range[:7]
        self.sim.data.ctrl[:7] = np.clip(self.sim.data.ctrl[:7], ctrlrange[:7, 0], ctrlrange[:7, 1])

        self.sim.data.ctrl[7:9] = gripper_ctrl

        # object actions
        action_obj = action[8:]
        action_obj = action_obj.reshape(self.n_objects, -1)
        
        obj_ctrl = np.concatenate((np.zeros((self.n_objects, 3)), 
                                    np.ones((self.n_objects, 1)), np.zeros((self.n_objects, 3))), axis=1)
        for i_action in range(len(self.obj_action_type)):
            obj_ctrl[:,self.obj_action_type[i_action]] = action_obj[:,i_action]

        if self.ai_object:
            obj_ctrl *= 0.05
        else:
            obj_ctrl *= 0.00

        obj_ctrl = np.concatenate((obj_ctrl, np.zeros((self.sim.model.nmocap-self.n_objects, 7))), axis=0)

        action_obj = np.concatenate([obj_ctrl.ravel()])

        utils.mocap_set_action(self.sim, action_obj)

    def _get_obs(self):

        obj_grp = self.max_n_objects if self.ai_object else 0

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs_all = []

        object_pos, object_rot, object_velp, object_velr, object_rel_pos = [], [], [], [], []
        for i_object in range(obj_grp, obj_grp + self.n_objects):
            # observations for the robot
            object_pos.append(self.sim.data.get_site_xpos('object' + str(i_object)))
            # rotations
            object_rot.append(rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i_object))))
            # velocities
            object_velp.append(self.sim.data.get_site_xvelp('object' + str(i_object)) * dt - grip_velp)
            object_velr.append(self.sim.data.get_site_xvelr('object' + str(i_object)) * dt)
            # gripper state
            object_rel_pos.append(self.sim.data.get_site_xpos('object' + str(i_object)) - grip_pos)
        
        object_pos = np.asarray(object_pos)
        object_rot = np.asarray(object_rot)
        object_velp = np.asarray(object_velp)
        object_velr = np.asarray(object_velr)
        object_rel_pos = np.asarray(object_rel_pos)
            
        obs = np.concatenate([
            grip_pos, 
            object_pos.ravel(), 
            object_rel_pos.ravel(), 
            gripper_state, 
            object_rot.ravel(),
            object_velp.ravel(), 
            object_velr.ravel(), 
            grip_velp, 
            gripper_vel
        ])
        if self.observe_obj_grp:
            obs = np.concatenate([obs, np.asarray([obj_grp])])

        obs_all.append(obs.copy())

        target_pos = self.goal.copy().reshape(self.n_objects,-1)
        target_rel_pos, target_velp = [], []
        for i_object in range(obj_grp, obj_grp + self.n_objects):
            # observations for the object
            # object state wrt target
            target_rel_pos.append(target_pos[i_object%self.max_n_objects] - self.sim.data.get_site_xpos('object' + str(i_object)))
            target_velp.append(0 - self.sim.data.get_site_xvelp('object' + str(i_object)) * dt)

        target_rel_pos = np.asarray(target_rel_pos)
        target_velp = np.asarray(target_velp)    

        obs = np.concatenate([
            np.zeros_like(grip_pos), 
            object_pos.ravel(), 
            target_rel_pos.ravel(), 
            np.zeros_like(gripper_state), 
            object_rot.ravel(),
            target_velp.ravel(), 
            object_velr.ravel(), 
            np.zeros_like(grip_velp), 
            np.zeros_like(gripper_vel)
        ])
        if self.observe_obj_grp:
            obs = np.concatenate([obs, np.asarray([self.max_n_objects])])

        obs_all.append(obs.copy())

        obs = np.asarray(obs_all)
        achieved_goal = object_pos.copy().ravel()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        target_pos = self.goal.copy().reshape(self.n_objects,-1)
        for i_target in range(0, self.n_objects):
            site_id = self.sim.model.site_name2id('target' + str(i_target))
            self.sim.model.site_pos[site_id] = target_pos[i_target] - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):

        self.deactivate_ai_object() 

        self.sim.set_state(self.initial_state)

        obj_grp = self.max_n_objects if self.ai_object else 0

        # First put all objects beside the table
        for i_object in range(0, self.max_n_objects*2):
            self.sim.data.set_joint_qpos('object' + str(i_object) + ':joint', self.initial_qpos['object' + str(i_object) + ':joint'])
        # Randomize start position of object.
        placement_count = 0
        for i_object in range(obj_grp, obj_grp + self.n_objects):
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                placement_count += 1
                for j_object in range(obj_grp, i_object):
                    if np.linalg.norm(object_xpos - self.sim.data.get_joint_qpos('object' + str(j_object) + ':joint')[:2]) < 0.070:
                        object_xpos = self.initial_gripper_xpos[:2]
                        break
                if placement_count >= 1000:
                    print('object placement is ended as maximum number of trials has been reached')
                    break
            object_qpos = self.sim.data.get_joint_qpos('object' + str(i_object) + ':joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self.height_offset 
            self.sim.data.set_joint_qpos('object' + str(i_object) + ':joint', object_qpos)

        self.sim.forward()

        for _ in range(10):
            self._set_action(np.zeros(self.n_actions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False

        if self.ai_object:
            self.activate_ai_object() 

        return True

    def _sample_goal(self):

        if self.target_stacked:
            goal = self.sim.data.get_joint_qpos('object0:joint')[:3]
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)

        goal_all = []
        goal_all.append(goal.copy())

        for  i_object in range(1, self.n_objects):
            if self.target_stacked:
                goal_all.append(goal.copy() + [0., 0., 0.05*i_object])
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
                goal_all.append(goal.copy())

        goal_all = np.asarray(goal_all)

        return goal_all.copy().ravel()

    def _is_success(self, achieved_goal, desired_goal):

        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):

        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]
        
        # Just to send object0 back to the floor 
        self.initial_qpos['object0:joint'] = [0.00, 0.025, 0.025, 1., 0., 0., 0.]

    def activate_ai_object_with_step(self):
        utils.activate_weld_eqs(self.sim)
        for _ in range(10):
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False

    def deactivate_ai_object_with_step(self):
        utils.deactivate_weld_eqs(self.sim)
        for _ in range(10):
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False

    def activate_ai_object(self):
        utils.activate_weld_eqs(self.sim)

    def deactivate_ai_object(self):
        utils.deactivate_weld_eqs(self.sim)
        

