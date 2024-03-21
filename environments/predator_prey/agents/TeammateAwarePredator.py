import math

from environments.predator_prey.agents import BasePredator
from environments.shared import NOOP, goto, compute_vicinity, pathfinding_distance


class TeammateAwarePredator(BasePredator):

    def __init__(self, agent_id: int, num_agents: int, num_preys: int, world_size: tuple) -> None:
        super().__init__(agent_id, num_agents, num_preys, world_size)

    def prey_selection_strategy(self, agent_locations, prey_locations, preys_alive):

        chosen_prey = None
        for prey_id, prey_alive in enumerate(preys_alive):
            if prey_alive:

                needs_final_predator = True
                prey_vicinity = compute_vicinity(prey_locations[prey_id])

                for agent_id, agent_location in enumerate(agent_locations):
                    if agent_id != self.agent_id and agent_location not in prey_vicinity:
                        needs_final_predator = False
                        break

                if needs_final_predator:
                    chosen_prey = prey_id
                    break

        return chosen_prey if chosen_prey is not None else self.closest_prey(agent_locations, prey_locations, preys_alive)

    def seek_strategy(self, prey_id, agent_locations, prey_locations, preys_alive):

        min_dist = math.inf
        closest_vacant_prey_vicinity_cell = None
        for prey_vicinity_cell in compute_vicinity(prey_locations[prey_id]):
            if prey_vicinity_cell not in agent_locations:
                dist = pathfinding_distance(self.agent_id, prey_vicinity_cell, agent_locations, prey_locations, preys_alive, self.world_size)
                if dist < min_dist:
                    min_dist = dist
                    closest_vacant_prey_vicinity_cell = prey_vicinity_cell

        if closest_vacant_prey_vicinity_cell is not None:
            action = goto(self.agent_id, prey_locations[prey_id], agent_locations, prey_locations, preys_alive, self.world_size)
        else:
            action = NOOP
        return action
