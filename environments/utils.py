import random
import numpy as np


def random_action(num_actions: int):
    return random.randrange(num_actions)


def sample_discrete_action(policy: np.ndarray, deterministic=False):
    if not deterministic:
        return np.random.choice(range(len(policy)), p=policy)
    else:
        argmaxes = np.argwhere(policy == np.max(policy)).reshape(-1)
        return random.choice(argmaxes)


def deterministic_policy(action: int, num_actions: int):
    policy = np.zeros((num_actions,))
    policy[action] = 1.0
    return policy

