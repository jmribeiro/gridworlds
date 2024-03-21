import math
from abc import abstractmethod, ABC
from typing import List, Tuple, Any

import numpy as np
from pfrl.agent import Agent
from scipy.spatial.distance import cityblock

from environments.shared import parse_grid_observation, NUM_ACTIONS, NORTH, SOUTH, EAST, WEST, NOOP, compute_vicinity, FORAGE
from environments.utils import sample_discrete_action

CAPTURE = FORAGE


class BasePredator(Agent, ABC):

    def __init__(self, agent_id: int, num_agents: int, num_preys: int, world_size: tuple) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_preys = num_preys
        self.world_size = world_size

    def policy(self, obs: Any) -> np.ndarray:

        obs: np.ndarray = obs
        if obs.dtype in [np.float32, np.float64]:
            obs = obs.astype(np.int32)

        policy = np.zeros(NUM_ACTIONS)

        agent_locations, prey_locations, _, _, preys_alive = parse_grid_observation(obs, self.num_agents, self.num_preys)
        location = agent_locations[self.agent_id]
        prey_id = self.prey_selection_strategy(agent_locations, prey_locations, preys_alive)

        if prey_id is None:
            actions = [NORTH, SOUTH, EAST, WEST, NOOP]
            prob = 1 / len(actions)
            for a in actions: policy[a] = prob
            return policy
        else:
            prey_location = prey_locations[prey_id]
            agent_vicinity = compute_vicinity(location)
            can_capture = prey_location in agent_vicinity
            action = CAPTURE if can_capture else self.seek_strategy(prey_id, agent_locations, prey_locations, preys_alive)
            policy[action] = 1.0
            return policy

    def prey_selection_strategy(self, agent_locations, prey_locations, preys_alive):
        return self.closest_prey(agent_locations, prey_locations, preys_alive)

    @abstractmethod
    def seek_strategy(self, prey_id, agent_locations, prey_locations, preys_alive):
        pass

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

    def closest_prey(self, agent_locations, prey_locations, preys_alive):
        min_dist = math.inf
        closest_prey = None
        for prey_id, prey_alive in enumerate(preys_alive):
            if prey_alive:
                dist = cityblock(agent_locations[self.agent_id], prey_locations[prey_id])
                if dist < min_dist:
                    closest_prey = prey_id
                    min_dist = dist
        return closest_prey
