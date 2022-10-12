import os
from gym import utils
from gym.envs.robotics import fetch_cage_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_cage.xml')


class FetchPushCageEnv(fetch_cage_env.FetchCageEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:shoulder_pan_joint': 0,
            'robot0:shoulder_lift_joint': -np.pi / 3,
            'robot0:upperarm_roll_joint': 0,
            'robot0:elbow_flex_joint': np.pi / 2,
            'robot0:forearm_roll_joint': 0,
            'robot0:wrist_flex_joint': np.pi / 4,
            'robot0:wrist_roll_joint': 0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_cage_env.FetchCageEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='dense')
        utils.EzPickle.__init__(self)

