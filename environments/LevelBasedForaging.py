import random
from collections import defaultdict

import gym
import numpy as np
from gym.spaces import Box

from environments.shared import NUM_ACTIONS, NOOP, NORTH, SOUTH, WEST, EAST, FORAGE, compute_vicinity, \
    MaxAppleSpawnAttemptException, parse_grid_observation

# Used when spawning apples (if two apples are placed close to one another preventing being foraged)
MAX_RESTART_ATTEMPTS = 200


class LevelBasedForaging(gym.Env):

    def __init__(
            self,
            num_agents, max_level, world_size, max_apples,
            always_max_apples=False, forage_mode="normal", failed_foraging_penalty=0.0
    ):

        if forage_mode not in ["normal", "cooperative", "competitive"]:
            raise ValueError(f"Invalid foraging model {forage_mode}")

        self.world_size = world_size
        self.num_agents = num_agents
        self.max_level = max_level
        self.num_apples = max_apples
        self.forage_mode = forage_mode
        self.failed_foraging_penalty = failed_foraging_penalty
        self.always_max_apples = always_max_apples

        # Current Episode
        self.agent_locations = None
        self.apple_locations = None
        self.agent_levels = None
        self.apple_levels = None
        self.active_apples = None
        self.terminal = False

        # Open AI Gym #
        agent_action_space = gym.spaces.Discrete(NUM_ACTIONS)
        observation_shape = (self.num_agents * 2 + self.num_apples * 2 + self.num_agents + self.num_apples + self.num_apples,)
        agent_observation_space = Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
        self.action_space = gym.spaces.Tuple(tuple([agent_action_space] * num_agents))
        self.observation_space = gym.spaces.Tuple(tuple([agent_observation_space] * num_agents))
        self.metadata = {"render.modes": ["human"]}
        self.viewer = None

    # ########## #
    # OpenAI Gym #
    # ########## #

    def reset(self):
        agent_locations, agent_levels, apple_locations, apple_levels, active_apples = self.restart()
        self.agent_locations = agent_locations
        self.apple_locations = apple_locations
        self.agent_levels = agent_levels
        self.apple_levels = apple_levels
        self.active_apples = active_apples
        self.terminal = False
        obs = self.default_features(
            agent_locations,
            apple_locations,
            agent_levels,
            apple_levels,
            active_apples
        )
        nobs = [obs for _ in range(self.num_agents)]
        return nobs

    def step(self, actions):
        self.agent_locations, self.active_apples, apples_foraged = self.transition(self.agent_locations, self.apple_locations, self.agent_levels, self.apple_levels, self.active_apples, actions)
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
            self.viewer = NonEnvViewer(self.world_size)
        return self.viewer.render(self.agent_locations, self.agent_levels, self.apple_locations, self.apple_levels, self.active_apples, return_rgb_array=True)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    # ##### #
    # Reset #
    # ##### #

    def restart(self):

        """
        Restarts the episode by generating random locations of the agents and apples.
        Returns:
        """

        done = False
        while not done:
            try:

                # 1 - Spawn agents
                agent_locations, agent_levels = self.spawn_agents()

                # 2 - Spawn apples
                sorted_levels = sorted(agent_levels)
                highest_four_levels = sorted_levels[-4:]    # Apples can only be foraged by four agents at a time
                max_forageable_level = sum(highest_four_levels)
                apple_locations, apple_levels, active_apples = self.spawn_apples(max_forageable_level, agent_locations, agent_levels)
                done = True

            except MaxAppleSpawnAttemptException:
                # Avoids staying stuck at spawning apples and tries again. See also self.spawn_apples
                done = False

        return agent_locations, agent_levels, apple_locations, apple_levels, active_apples

    def spawn_agents(self):
        """
        Returns:
            Two lists,
            (i) agent_locations, a list containing the (row, column) of each agent and
            (ii) agent_levels, a list containing the level of each int
        """
        agent_locations = []
        agent_levels = []
        for a in range(self.num_agents):
            agent_level = random.randint(1, self.max_level)
            agent_location = None
            while agent_location is None or agent_location in agent_locations:
                agent_location = random.randint(0, self.world_size[0] - 1), random.randint(0, self.world_size[1] - 1)
            agent_locations.append(agent_location)
            agent_levels.append(agent_level)
        return agent_locations, agent_levels

    def spawn_apples(self, max_forageable_level, agent_locations, agent_levels):

        """
        Returns:
            Two lists,
            (i) apple_locations, a list containing the (row, column) of each apple and
            (ii) apple_levels, a list containing the level of each apple
        """
        _apple_locations = []
        _apple_levels = []
        _apples_forageable = []
        num_starting_active_apples = random.randint(1, self.num_apples) if not self.always_max_apples else self.num_apples
        for apple_id in range(self.num_apples):

            if self.forage_mode == "normal": apple_level = random.randint(1, max_forageable_level)
            elif self.forage_mode == "cooperative": apple_level = max_forageable_level
            elif self.forage_mode == "competitive": apple_level = min(agent_levels)
            else: raise ValueError("Invalid mode")

            if apple_id < num_starting_active_apples:

                apple_location = None
                other_apples_in_vicinity = False

                attempts = 0
                while \
                        apple_location is None or \
                        apple_location in agent_locations or \
                        apple_location in _apple_locations or \
                        other_apples_in_vicinity:

                    if attempts > MAX_RESTART_ATTEMPTS:
                        raise MaxAppleSpawnAttemptException()

                    # Start at 1 and go to border - 2 to avoid spawning apples in places that are not able to circle around
                    apple_location = random.randint(1, self.world_size[0] - 2), random.randint(1, self.world_size[1] - 2)
                    apple_vicinity = compute_vicinity(apple_location)
                    other_apples_in_vicinity = any([other_apple_location in apple_vicinity for other_apple_location in _apple_locations])
                    attempts += 1

                apple_forageable = True

            else:
                apple_location = random.randint(1, self.world_size[0] - 2), random.randint(1, self.world_size[1] - 2)
                apple_level = apple_level
                apple_forageable = False

            _apple_locations.append(apple_location)
            _apple_levels.append(apple_level)
            _apples_forageable.append(apple_forageable)

        random_order = list(range(self.num_apples))
        random.shuffle(random_order)
        apple_locations, apple_levels, apples_forageable = [], [], []
        for i in random_order:
            apple_locations.append(_apple_locations[i])
            apple_levels.append(_apple_levels[i])
            apples_forageable.append(_apples_forageable[i])

        return apple_locations, apple_levels, apples_forageable

    # #### #
    # Step #
    # #### #

    def dynamics_fn(self, grid_observation, actions):
        num_agents = self.num_agents
        num_apples = self.num_apples   # Ask me and I'll explain
        agent_locations, apple_locations, agent_levels, apple_levels, active_apples = parse_grid_observation(grid_observation, num_agents, num_apples)
        agent_locations, active_apples, apples_foraged = self.transition(agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions)
        next_grid_observation = self.default_features(agent_locations, apple_locations, agent_levels, apple_levels, active_apples)
        rewards = self.reward(actions, apples_foraged, apple_levels)
        apples_remaining = sum(active_apples)
        terminal = apples_remaining == 0
        return next_grid_observation, rewards, terminal

    def transition(self, agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions):
        next_agent_locations, agents_attempting_foraging, apples_being_foraged = self.process_actions(agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions)
        apples_foraged = self.process_foraging(agents_attempting_foraging, apples_being_foraged, apple_locations, agent_levels, apple_levels)
        next_active_apples = [(active_apples[apple_id] if apple_id not in apples_foraged else False) for apple_id in range(self.num_apples)]
        return next_agent_locations, next_active_apples, apples_foraged

    def reward(self, actions, apples_foraged, apple_levels):
        rewards = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            apple_level, num_foragers = self.check_if_agent_foraged(agent_id, apples_foraged, apple_levels)
            agent_foraged_apple = apple_level is not None
            agent_still_tried_to_forage = actions[agent_id] == FORAGE
            if agent_foraged_apple:
                rewards[agent_id] += apple_level / num_foragers
            elif agent_still_tried_to_forage:
                rewards[agent_id] -= self.failed_foraging_penalty
        return rewards

    def process_actions(self, agent_locations, apple_locations, agent_levels, apple_levels, active_apples, actions):
        agent_acting_order = list(range(self.num_agents))
        random.shuffle(agent_acting_order)
        apples_being_foraged = np.zeros(self.num_apples)
        agents_attempting_foraging = defaultdict(lambda: [])
        next_agent_locations = dict([(agent_id, agent_locations[agent_id]) for agent_id in range(self.num_agents)])
        for agent_id in agent_acting_order:
            action = actions[agent_id]
            self.process_agent_action(agent_id, action, next_agent_locations, apple_locations, agent_levels, apple_levels, active_apples, apples_being_foraged, agents_attempting_foraging)
        next_agent_locations = [next_agent_locations[a] for a in range(self.num_agents)]
        return next_agent_locations, agents_attempting_foraging, apples_being_foraged

    def process_agent_action(self, agent_id, action, next_agent_locations, apple_locations, agent_levels, apple_levels, active_apples, apple_being_foraged, agents_foraging):

        agent_level = agent_levels[agent_id]
        noop = action == NOOP
        moving = action != FORAGE and action != NOOP
        foraging = action == FORAGE

        if not noop:

            agent_location = next_agent_locations[agent_id]

            if moving:
                next_location = self.compute_next_agent_location(agent_location, action)
                agent_collision = next_location in [next_agent_locations[other_agent_id] for other_agent_id in range(self.num_agents) if other_agent_id != agent_id]
                apple_collision = next_location in [apple_locations[apple_id] for apple_id in range(self.num_apples) if active_apples[apple_id]]
                collision = agent_collision or apple_collision
                next_agent_locations[agent_id] = agent_location if collision else next_location

            elif foraging:
                agent_vicinity = compute_vicinity(agent_location)
                for apple_id, apple_location in enumerate(apple_locations):
                    apple_is_active = active_apples[apple_id]
                    if apple_location in agent_vicinity and apple_is_active:
                        apple_being_foraged[apple_id] += agent_level
                        agents_foraging[agent_id].append(apple_id)

            else:
                raise ValueError("Should be unreachable (Invalid action)")

    def compute_forageable_apples(self, apple_locations, apple_levels, apples_being_foraged):
        active_apple_levels = [apple_levels[apple_id] for apple_id, apple_location in enumerate(apple_locations) if apple_location is not None]
        apples_forageable = {}
        max_level = max(active_apple_levels)
        for apple_level in range(1, max_level + 1):
            apples_forageable[apple_level] = []
        for apple_id in range(self.num_apples):
            apple_level = apple_levels[apple_id]
            if apple_level is not None:
                level_applied_to_apple = apples_being_foraged[apple_id]
                can_be_foraged = level_applied_to_apple >= apple_level
                if can_be_foraged:
                    apples_forageable[apple_level].append(apple_id)
        return apples_forageable

    def compute_apples_that_can_be_foraged(self, agents_attempting_foraging, apples_forageable_by_level):
        apples_that_can_be_foraged = []
        for apple_level in apples_forageable_by_level:
            apples = apples_forageable_by_level[apple_level]
            for apple_id in apples:
                for agent_id in agents_attempting_foraging:
                    apples_being_foraged_by_agent = agents_attempting_foraging[agent_id]
                    if apple_id in apples_being_foraged_by_agent:
                        apples_that_can_be_foraged.append(apple_id)
        return apples_that_can_be_foraged

    def process_foraging(self, agents_attempting_foraging, apples_being_foraged, apple_locations, agent_levels, apple_levels):

        apples_foraged = {}
        if len(agents_attempting_foraging) > 0:

            apples_forageable_by_level = self.compute_forageable_apples(apple_locations, apple_levels, apples_being_foraged)

            if any([len(apples_forageable_by_level[apple_level]) > 0 for apple_level in apples_forageable_by_level]):

                apples_that_can_be_foraged = self.compute_apples_that_can_be_foraged(agents_attempting_foraging, apples_forageable_by_level)
                there_are_apples_that_can_be_foraged = len(apples_that_can_be_foraged) > 0

                step = 0
                while there_are_apples_that_can_be_foraged:

                    step += 1
                    for apple_level in reversed(apples_forageable_by_level):

                        apple_candidates = apples_forageable_by_level[apple_level]

                        if len(apple_candidates) > 0:

                            apple_id = random.choice(apple_candidates)

                            agents_that_foraged_apple = []
                            level_required = apple_level
                            for agent_id in agents_attempting_foraging:
                                if apple_id in agents_attempting_foraging[agent_id]:
                                    level_required -= agent_levels[agent_id]
                                    agents_that_foraged_apple.append(agent_id)

                            if level_required <= 0:

                                # Apple successfully foraged
                                # print(f"Apple {apple_id} foraged by {agents_that_foraged_apple}")
                                apples_foraged[apple_id] = agents_that_foraged_apple

                                for agent_id in agents_that_foraged_apple:
                                    if agent_id in agents_attempting_foraging:
                                        del agents_attempting_foraging[agent_id]

                    apples_that_can_be_foraged = self.compute_apples_that_can_be_foraged(agents_attempting_foraging, apples_forageable_by_level)
                    there_are_apples_that_can_be_foraged = len(apples_that_can_be_foraged) > 0

        return apples_foraged

    def check_if_agent_foraged(self, agent_id, apples_foraged, apple_levels):
        for apple_id, foragers in apples_foraged.items():
            apple_level = apple_levels[apple_id]
            num_foragers = len(foragers)
            if agent_id in foragers:
                return apple_level, num_foragers
        return None, 0

    def compute_next_agent_location(self, agent_location, action):

        row_border = self.world_size[0] - 1
        column_border = self.world_size[1] - 1

        if action == NORTH:
            next_location = [
                max(0, agent_location[0] - 1),
                agent_location[1]
            ]
        elif action == SOUTH:
            next_location = [
                min(row_border, agent_location[0] + 1),
                agent_location[1]
            ]

        elif action == WEST:
            next_location = [
                agent_location[0],
                max(0, agent_location[1] - 1),
            ]
        elif action == EAST:
            next_location = [
                agent_location[0],
                min(column_border, agent_location[1] + 1),
            ]
        else:
            next_location = agent_location

        return tuple(next_location)

    @staticmethod
    def default_features(agent_locations, apple_locations, agent_levels, apple_levels, active_apples):
        locations = np.array(agent_locations + apple_locations).reshape(-1)
        levels = np.array(agent_levels + apple_levels)
        active_apples = np.array(active_apples)
        obs = np.concatenate((locations, levels, active_apples))
        return obs
