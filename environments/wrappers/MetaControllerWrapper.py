import itertools

import numpy as np
from gym import Wrapper
from gym.spaces import Box, Discrete


class MetaControllerWrapper(Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.num_agents = len(env.observation_space)
        self.observation_spaces = env.observation_space
        self.action_spaces = env.action_space

        features = sum([self.observation_spaces[agent_id].shape[0] for agent_id in range(self.num_agents)])
        shape = (features, )
        self.observation_space = Box(-np.inf, np.inf, shape)
        self.joint_actions = list(itertools.product(*[list(range(space.n)) for space in self.action_spaces]))
        self.action_space = Discrete(len(self.joint_actions))

    def reset(self, **kwargs):
        nobs = super().reset(**kwargs)
        meta_obs = np.stack(nobs).reshape(-1)
        return meta_obs

    def step(self, action):
        actions = self.joint_actions[action]
        next_nobs, rewards, terminals, info = super().step(actions)
        meta_next_obs = np.stack(next_nobs).reshape(-1)
        reward = sum(rewards)
        terminal = all(terminals)
        return meta_next_obs, reward, terminal, info
