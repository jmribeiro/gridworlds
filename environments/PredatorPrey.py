import random
from environments import LevelBasedForaging
from environments.shared import parse_grid_observation


class PredatorPrey(LevelBasedForaging):

    def __init__(
            self,
            num_agents, world_size, max_preys, capture_mode="cooperative", failed_capture_penalty=0.0
    ):
        super(PredatorPrey, self).__init__(num_agents, 1, world_size, max_preys, True, capture_mode, failed_capture_penalty)

    # ########## #
    # OpenAI Gym #
    # ########## #

    def step(self, actions):
        self.agent_locations, self.apple_locations, self.active_apples, apples_foraged = self.transition(self.agent_locations, self.apple_locations, self.agent_levels, self.apple_levels, self.active_apples, actions)
        obs = self.default_features(
            self.agent_locations,
            self.apple_locations,
            self.agent_levels,
            self.apple_levels,
            self.active_apples
        )
        nobs = [obs for _ in range(self.num_agents)]
        rewards = self.reward(actions, apples_foraged, self.apple_levels)
        apples_remaining = sum(self.active_apples)
        terminal = apples_remaining == 0
        return nobs, rewards.tolist(), [terminal for _ in range(self.num_agents)], {}

    def render(self, mode="human"):
        if self.viewer is None:
            from environments.shared.rendering import NonEnvViewer
            self.viewer = NonEnvViewer(self.world_size, "../predator_prey/icons")
        return self.viewer.render(self.agent_locations, self.agent_levels, self.apple_locations, self.apple_levels, self.active_apples, return_rgb_array=True)

    # #### #
    # Step #
    # #### #

    def dynamics_fn(self, grid_observation, actions):
        num_agents = self.num_agents
        num_apples = self.num_apples   # Ask me and I'll explain
        agent_locations, apple_locations, agent_levels, apple_levels, active_apples = parse_grid_observation(grid_observation, num_agents, num_apples)
        agent_locations, apple_locations, active_apples, apples_foraged = self.transition(agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions)
        next_grid_observation = self.default_features(agent_locations, apple_locations, agent_levels, apple_levels, active_apples)
        rewards = self.reward(actions, apples_foraged, apple_levels)
        apples_remaining = sum(active_apples)
        terminal = apples_remaining == 0
        return next_grid_observation, rewards, terminal

    def transition(self, agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions):
        next_agent_locations, agents_attempting_foraging, apples_being_foraged = self.process_actions(agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions)
        apples_foraged = self.process_foraging(agents_attempting_foraging, apples_being_foraged, apple_locations, agent_levels, apple_levels)
        next_active_apples = [(active_apples[apple_id] if apple_id not in apples_foraged else False) for apple_id in range(self.num_apples)]
        next_apple_locations = []
        for apple_id, next_active_apple in enumerate(next_active_apples):
            next_apple_location = apple_locations[apple_id]
            if next_active_apple:
                random_action = random.randrange(4) # UP, DOWN, LEFT, RIGHT
                previous_apple_location = next_apple_location
                next_apple_location = self.compute_next_agent_location(next_apple_location, random_action)
                collision = next_apple_location in next_agent_locations or next_apple_location in next_apple_locations
                if collision or not self.cell_inside(next_apple_location):
                    next_apple_location = previous_apple_location
            next_apple_locations.append(next_apple_location)
        return next_agent_locations, next_apple_locations, next_active_apples, apples_foraged

    def cell_inside(self, cell):
        row, col = cell
        return row != 0 and col != 0 and row != self.world_size[0] - 1 and col != self.world_size[1] - 1