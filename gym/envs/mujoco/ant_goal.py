import numpy as np
import random
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
import pdb

class AntEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.n_tasks = 4
        self.tasks = self.sample_tasks(self.n_tasks)
        self.tasks[0] = np.array([-10, 0])
        self.tasks[1] = np.array([5, 5])
        self.tasks[2] = np.array([-5, 5])
        self.tasks[3] = np.array([-5, -5])
        self.set_task(self.tasks[0])
        self.t = 0
        self.reward_type = 'dense'
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        a = np.random.random(n_tasks) * 2 * np.pi
        r = 3 * np.random.random(n_tasks) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_pos = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_pos

    def step(self, a):
        # if self.t == 250:
        #     self.set_task(self.tasks[1])
        # if self.t == 500:
        #     self.set_task(self.tasks[2])
        # if self.t == 750:
        #     self.set_task(self.tasks[3])
        
        posbefore = np.copy(self.get_body_com("torso"))
        self.do_simulation(a, self.frame_skip)
        posafter = self.get_body_com("torso")
        if self.reward_type == 'dense':
            reward = -np.linalg.norm(self.get_task() - posafter[:2])
        else:
            if np.linalg.norm(self.get_task() - posafter[:2]) <= 1:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        ctrl_cost = contact_cost = survive_reward = 0
        done=False
        ob = self._get_obs()
        self.t += 1
        return ob, reward, done, dict(
            reward_forward=reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat, 
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.t = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == "__main__":
    env = AntGoalEnv()
    while True:
        env.reset()
        for _ in range(100):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action