import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

from environments.shared import ACTION_MEANINGS


class SingleAgentAdapter(Env):

    def __init__(self, ma_env: Env, agent_id: int, agents: dict, observation_fn: callable, observation_shape: tuple, teammates_actions: bool):

        self.agent_id = agent_id
        self.ma_env = ma_env
        self.agents = agents
        self.num_agents = len(agents) + 1
        self.view_teammates_actions = teammates_actions

        self.obs_n = [None for _ in range(self.num_agents)]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=observation_shape)
        self.action_space = Discrete(ma_env.action_space[self.agent_id].n-1) # Drop NOOP for
        self.reward_range = ma_env.reward_range[self.agent_id]
        self.observation_fn = observation_fn
        self.policy_fns = lambda obs: [agent.policy(obs) for _, agent in self.agents.items()]
        self.action_meanings = ACTION_MEANINGS[:-1]

    def reset(self, **kwargs):
        self.obs_n = self.ma_env.reset()
        return self.observation_fn(self.obs_n[self.agent_id])

    def step(self, action):
        actions = [
            action if agent_id == self.agent_id else self.agents[agent_id].act(self.obs_n[agent_id])
            for agent_id in range(self.num_agents)
        ]
        self.obs_n, rewards, terminals, info = self.ma_env.step(actions)
        info["actions"] = actions

        next_obs = self.observation_fn(self.obs_n[self.agent_id])
        if self.view_teammates_actions:
            teammates_actions = np.array([action for a, action in enumerate(actions) if a != self.agent_id])
            next_obs = np.concatenate((next_obs, teammates_actions))

        return next_obs, rewards[self.agent_id], terminals[self.agent_id], info

    def dynamics_fn(self, features, action):
        actions = [
            action if agent_id == self.agent_id else self.agents[agent_id].act(features)
            for agent_id in range(self.num_agents)
        ]
        next_features, rewards, terminal = self.ma_env.dynamics_fn(features, actions)
        return next_features, rewards[self.agent_id], terminal

    def render(self, mode="human"):
        self.ma_env.render(mode)
