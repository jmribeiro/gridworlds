import math
from scipy.spatial.distance import cityblock

from environments.predator_prey.agents import BasePredator
from environments.shared import NOOP, compute_vicinity, goto, greedy_direction


class GreedyPredator(BasePredator):

    def __init__(self, agent_id: int, num_agents: int, num_preys: int, world_size: tuple) -> None:
        super().__init__(agent_id, num_agents, num_preys, world_size)

    def seek_strategy(self, prey_id, agent_locations, prey_locations, preys_alive):

        min_dist = math.inf
        free_vicinity_cells = []
        closest_vacant_prey_vicinity_cell = None
        for prey_vicinity_cell in compute_vicinity(prey_locations[prey_id]):
            if prey_vicinity_cell not in agent_locations:
                free_vicinity_cells.append(free_vicinity_cells)
                dist = cityblock(prey_vicinity_cell, agent_locations[self.agent_id])
                if dist < min_dist:
                    min_dist = dist
                    closest_vacant_prey_vicinity_cell = prey_vicinity_cell

        if len(free_vicinity_cells) == 1:
            action = goto(self.agent_id, prey_locations[prey_id], agent_locations, prey_locations, preys_alive, self.world_size)
        elif closest_vacant_prey_vicinity_cell is not None:
            action = greedy_direction(agent_locations[self.agent_id], closest_vacant_prey_vicinity_cell)
        else:
            action = NOOP

        return action
