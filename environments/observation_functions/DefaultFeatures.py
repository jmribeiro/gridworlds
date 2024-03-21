class DefaultFeatures:

    def __init__(self, num_agents, num_apples):
        self.shape = (num_agents * 2 + num_apples * 2 + num_agents + num_apples + num_apples,)

    def __call__(self, obs):
        return obs
