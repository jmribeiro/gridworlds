from typing import List, Tuple, Any

import numpy as np
from pfrl.agent import Agent

from environments.shared import parse_grid_observation, find_highest_level_apple, goto, compute_vicinity, NOOP, WEST, EAST, SOUTH, NORTH, FORAGE, NUM_ACTIONS
from environments.utils import sample_discrete_action


class MaxLevelForager(Agent):

    def __init__(self, agent_id: int, num_agents: int, num_apples: int, world_size: tuple) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_apples = num_apples
        self.world_size = world_size

    def policy(self, obs: Any) -> np.ndarray:

        obs: np.ndarray = obs
        if obs.dtype in [np.float32, np.float64]:
            obs = obs.astype(np.int32)

        policy = np.zeros(NUM_ACTIONS)

        agent_locations, apple_locations, agent_levels, apple_levels, active_apples = parse_grid_observation(obs, self.num_agents, self.num_apples)
        location, level = agent_locations[self.agent_id], agent_levels[self.agent_id]

        apple_id = find_highest_level_apple(location, apple_locations, apple_levels, active_apples)
        if apple_id is None:
            actions = [NORTH, SOUTH, EAST, WEST, NOOP] if self.agent_id != 0 else [NORTH, SOUTH, EAST, WEST]
            prob = 1 / len(actions)
            for a in actions: policy[a] = prob
            return policy
        else:
            apple_location = apple_locations[apple_id]
            agent_vicinity = compute_vicinity(location)
            if apple_location in agent_vicinity:
                policy[FORAGE] = 1.0
                return policy
            else:
                action = goto(self.agent_id, apple_location, agent_locations, apple_locations, active_apples, self.world_size)
                policy[action] = 1.0
                return policy

    def act(self, obs: Any) -> Any:
        policy = self.policy(obs)
        action = sample_discrete_action(policy)
        return action

    def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
        pass

    def save(self, dirname: str) -> None:
        pass

    def load(self, dirname: str) -> None:
        pass

    def get_statistics(self) -> List[Tuple[str, Any]]:
        pass
