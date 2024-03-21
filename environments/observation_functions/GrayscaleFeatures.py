import cv2

from environments.shared import parse_grid_observation


class GrayscaleFeatures:

    def __init__(self, num_agents, num_apples, world_size, width=84, height=84):
        from environments.shared.rendering import NonEnvViewer
        self.num_agents = num_agents
        self.num_apples = num_apples
        self.width = width
        self.height = height
        self.shape = (1, self.width, self.height)
        self.viewer = NonEnvViewer(world_size, hide_window=True)

    def __call__(self, obs):
        agent_locations, apple_locations, agent_levels, apple_levels, active_apples = parse_grid_observation(obs, self.num_agents, self.num_apples)
        rgb_array = self.viewer.render(agent_locations, agent_levels, apple_locations, apple_levels, active_apples, return_rgb_array=True)
        grayscale_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        resized_grayscale_array = cv2.resize(grayscale_array, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized_grayscale_array.reshape((1, self.width, self.height))
