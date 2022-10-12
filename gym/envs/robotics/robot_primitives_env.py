import os
import copy
import numpy as np
import pdb
import imageio
import IPython
import base64

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 512

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

class RobotPrimitivesEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, action_max):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-action_max, action_max, shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.x_borders = [1.1, 1.5]
        self.y_borders = [0.4, 1.0] 
        self.offset = action_max
        self.x_conversion = (self.x_borders[1] - self.x_borders[0]) / (2*action_max)
        self.y_conversion = (self.y_borders[1] - self.y_borders[0]) / (2*action_max)
        self.x_translation = 1
        self.y_translation = 1
        self.push_height = 0.45
        self.lift_height = 0.7 
        self.path_len = 3
        self.t = 0
        self.randomize_object = True

        self.recent_frames = []


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_waypoints(self, start_x, start_y, end_x, end_y, steps=3):
        delta_y = end_y - start_y
        delta_x = end_x - start_x
        waypoints = []
        for i in range(steps+1):
            waypoints.append(np.array([start_x + i*delta_x / steps, start_y + i*delta_y / steps]))
        return waypoints
        
    def step(self, action, make_video=False):
        obs_x_before = np.copy(self.sim.data.get_site_xpos('object0')[0])
        obs_y_before = np.copy(self.sim.data.get_site_xpos('object0')[1])

        self.phase = 'start'
        if action[0] >= 0: 
            if action[1] >= 0:
                start_x = self.sim.data.get_site_xpos('object0')[0] + 0.1
                end_x = self.sim.data.get_site_xpos('object0')[0] - 0.1
                start_y = self.sim.data.get_site_xpos('object0')[1] 
                end_y = self.sim.data.get_site_xpos('object0')[1]  
            else:
                start_x = self.sim.data.get_site_xpos('object0')[0] - 0.1
                end_x = self.sim.data.get_site_xpos('object0')[0] + 0.1
                start_y = self.sim.data.get_site_xpos('object0')[1] 
                end_y = self.sim.data.get_site_xpos('object0')[1] 
        else: 
            if action[1] >= 0:
                start_x = self.sim.data.get_site_xpos('object0')[0] 
                end_x = self.sim.data.get_site_xpos('object0')[0] 
                start_y = self.sim.data.get_site_xpos('object0')[1] - 0.1
                end_y = self.sim.data.get_site_xpos('object0')[1] + 0.1
            else:
                start_x = self.sim.data.get_site_xpos('object0')[0] 
                end_x = self.sim.data.get_site_xpos('object0')[0] 
                start_y = self.sim.data.get_site_xpos('object0')[1] + 0.1
                end_y = self.sim.data.get_site_xpos('object0')[1] - 0.1

        start_x = np.clip(start_x, self.x_borders[0], self.x_borders[1])
        start_y = np.clip(start_y, self.y_borders[0], self.y_borders[1])    
        end_x = np.clip(end_x, self.x_borders[0], self.x_borders[1])
        end_y = np.clip(end_y, self.y_borders[0], self.y_borders[1])

        # gripper
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0)
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0)
  
        if self.phase == 'start':
            n_loops = 0
            while np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip')[:3] - np.array([start_x, start_y, self.push_height])) > 0.05:
                gripper_target = np.array([start_x, start_y, self.push_height])
                gripper_rotation = np.array([1., 0., 1., 0.])
                self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                self.sim.step()
                if make_video: self.recent_frames.append(self.render_single())
                n_loops += 1

                
                if n_loops > 100:
                    print("Invalid start position. Resetting...")
                    return self._get_obs(), 0, True, {}
            self.phase = 'push'
        
        waypoints = self.compute_waypoints(start_x, start_y, end_x, end_y)
        if self.phase == 'push':
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0)
            for point in waypoints:
                gripper_target = np.array([point[0], point[1], self.push_height]) 
                gripper_rotation = np.array([1., 0., 1., 0.])
                self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                self.sim.step()
                if make_video: self.recent_frames.append(self.render_single())
            self.phase = 'lift'
        
        if self.phase == 'lift':
            while np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip')[2] - self.lift_height) > 0.05:
                gripper_target = [self.sim.data.get_site_xpos('robot0:grip')[0], 
                    self.sim.data.get_site_xpos('robot0:grip')[1],
                    self.lift_height]
                gripper_rotation = np.array([1., 0., 1., 0.])
                self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                self.sim.step()
                if make_video: self.recent_frames.append(self.render_single())
        
        obs = self._get_obs()

        done = False
        info={}
        reward = 1
        self.t += 1
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotPrimitivesEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        self.push = False
        obs = self._get_obs()
        self.recent_frames = []
        self.recent_frames.append(self.render_single())
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        frames = self.recent_frames
        self.recent_frames = []
        return frames

    def render_single(self, mode='rgb_array', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
