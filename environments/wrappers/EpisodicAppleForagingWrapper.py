from gym import Wrapper


class EpisodicAppleForagingWrapper(Wrapper):

    # Problem: If two apples are foraged simultaneously, reward will be the same (1.0)

    def __init__(self, env, num_agents, num_apples):
        super().__init__(env)
        self.current_active_apples = 0
        self.num_agents = num_agents
        self.num_apples = num_apples

    def step(self, action):
        next_obs, reward, terminal, info = super().step(action)
        active_apples = self.env.ma_env.active_apples
        num_active_apples = sum(active_apples)
        if num_active_apples < self.current_active_apples:
            # Apple foraged
            reward = 1.0
            terminal = True
            self.current_active_apples = num_active_apples
            self.fake_reset_obs = next_obs
        else:
            reward = 0.0
            terminal = False
            self.current_active_apples = num_active_apples
        true_terminal = num_active_apples == 0

        return next_obs, reward, terminal, info

    def reset(self, **kwargs):
        if self.current_active_apples == 0 or not hasattr(self, "fake_reset_obs"):
            obs = super().reset(**kwargs)
            active_apples = self.env.ma_env.active_apples
            self.current_active_apples = sum(active_apples)
            if hasattr(self, "fake_reset_obs"): del self.fake_reset_obs
            return obs
        else:
            return self.fake_reset_obs
