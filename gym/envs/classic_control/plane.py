import gym
from gym import spaces
import numpy as np

class PlaneEnv(gym.Env):
  """2D continuous plane Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(PlaneEnv, self).__init__()
    self.reward_range = (-10, 10)
    self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]))
    self.observation_space = spaces.Box(low=np.array([-100, -100]), high=np.array([100, 100]))
    self.goal_pos = np.array([10, 0])
    self.reward_type = 'dense'

  def step(self, action):
    posbefore = np.copy(self._obs)
    self._obs = self._obs + action
    posafter = self._obs
    obs = self._obs
    if self.reward_type == 'sparse':
      if np.linalg.norm(posafter - self.goal_pos) <= 1:
        reward = 1.0
        done = True
      else:
        reward = 0.0
        done = False
    else:
      reward = -np.linalg.norm(self.goal_pos - posafter)
      done = False
    self.t += 1
    if self.t == 50: self.goal_pos = np.array([-10, 0])
    return obs, reward, done, {}

  def reset(self):
    self._obs = np.array([0, 0]) 
    self.t = 0
    return self._obs
  
  def render(self, mode='human', close=False):
    if len(self._obs.shape) > 1:
      print("{} {}".format(self._obs[:, 0], self._obs[:, 1]))
    else:
      print("{} {}".format(self._obs[0], self._obs[1]))
