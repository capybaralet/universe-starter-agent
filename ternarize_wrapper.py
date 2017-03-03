import gym
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TernarizeWrapper(gym.Wrapper):
    """
    Replaces rewards with their signs
    """
    def __init__(self, env):
        super(TernarizeWrapper, self).__init__(env)
        #self.__dict__.update(locals())

    def _step(self, action):
        obs, reward, done, info = self.env.step(basic_action)
        return obs, np.sign(reward), done, info

