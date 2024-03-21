import numpy as np
from environments.shared import parse_grid_observation


def center_for(view_grid_size):
    return int((view_grid_size - 1) / 2)


class FieldOfViewChannels:

    def __init__(self, agent_id: int, world_size: tuple, num_agents: int, num_apples: int, view_grid_size: int, skip_levels: bool=False):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_apples = num_apples
        self.world_size = world_size
        self.view_grid_size = view_grid_size
        self.view_grid_relative_center = center_for(view_grid_size)
        self.channels = [f"teammate{a+1}" for a in range(num_agents-1)] + [f"apple{a}" for a in range(num_apples)] + ["walls"]
        if not skip_levels: self.channels += ["levels"]
        self.shape = (len(self.channels), view_grid_size, view_grid_size)

    def __call__(self, obs):

        agent_locations, apple_locations, agent_levels, apple_levels, active_apples = parse_grid_observation(obs, self.num_agents, self.num_apples)
        self_location = agent_locations[self.agent_id]
        self_row, self_column = self_location

        agent_channels = np.zeros((self.num_agents-1, self.view_grid_size, self.view_grid_size))
        apple_channels = np.zeros((self.num_apples, self.view_grid_size, self.view_grid_size))
        levels_channel = np.zeros((1, self.view_grid_size, self.view_grid_size))

        walls_channel = self.compute_walls(self_row, self_column).reshape((1, self.view_grid_size, self.view_grid_size))

        for agent_id, (row, column) in enumerate(agent_locations):
            teammate = agent_id != self.agent_id
            rel_row, rel_column = row - self_row, column - self_column
            if teammate and self.can_observe(rel_row, rel_column):
                view_grid_row, view_grid_col = self.view_grid_relative_center + rel_row, self.view_grid_relative_center + rel_column
                agent_channels[agent_id-1, view_grid_row, view_grid_col] = 1
                levels_channel[0, view_grid_row, view_grid_col] = agent_levels[agent_id]

        for apple_id, (row, column) in enumerate(apple_locations):
            active_apple = active_apples[apple_id]
            rel_row, rel_column = row - self_row, column - self_column
            if active_apple and self.can_observe(rel_row, rel_column):
                view_grid_row, view_grid_col = self.view_grid_relative_center + rel_row, self.view_grid_relative_center + rel_column
                apple_channels[apple_id, view_grid_row, view_grid_col] = 1
                levels_channel[0, view_grid_row, view_grid_col] = apple_levels[apple_id]

        channels = [agent_channels, apple_channels, walls_channel]
        if "levels" in self.channels: channels += [levels_channel]

        channels = np.concatenate(channels)
        obs = channels

        return obs

    def can_observe(self, rel_row, rel_column):
        min, max = -self.view_grid_relative_center, self.view_grid_relative_center
        in_bounds = min <= rel_row <= max and min <= rel_column <= max
        return in_bounds

    def compute_walls(self, self_row, self_col):
        walls = np.zeros((self.view_grid_size, self.view_grid_size))
        min, max = -self.view_grid_relative_center, self.view_grid_relative_center
        wall_coordinates = [
            (fov_row, fov_col)
            for fov_row, rel_row in enumerate(range(min, max+1))
            for fov_col, rel_col in enumerate(range(min, max+1))
            if not (0 <= self_row + rel_row < self.world_size[0]) or not (0 <= self_col + rel_col < self.world_size[1])
        ]
        if len(wall_coordinates) > 0:
            wall_rows, wall_cols = zip(*wall_coordinates)
            walls[wall_rows, wall_cols] = 1

        return walls

    def compute_walls_slow(self, self_row, self_col):
        walls = np.zeros((self.view_grid_size, self.view_grid_size))
        min, max = -self.view_grid_relative_center, self.view_grid_relative_center
        for fov_row, rel_row in enumerate(range(min, max+1)):
            for fov_col, rel_col in enumerate(range(min, max+1)):
                abs_row = self_row + rel_row
                abs_column = self_col + rel_col
                row_in_bounds = 0 <= abs_row < self.world_size[0]
                col_in_bounds = 0 <= abs_column < self.world_size[1]
                wall = int(not row_in_bounds or not col_in_bounds)
                walls[fov_row, fov_col] = wall
        return walls
