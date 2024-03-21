import math
from collections import defaultdict

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.spatial.distance import cityblock
from heapq import *

ACTION_MEANINGS = ["up", "down", "left", "right", "forage", "noop"]
NUM_ACTIONS = len(ACTION_MEANINGS)
NORTH, SOUTH, WEST, EAST, FORAGE, NOOP = range(NUM_ACTIONS)


class MaxAppleSpawnAttemptException(Exception):
    pass


def compute_vicinity(location):
    location_vec = np.array(location)
    north = tuple(location_vec + np.array([-1, 0]))
    south = tuple(location_vec + np.array([1, 0]))
    west = tuple(location_vec + np.array([0, -1]))
    east = tuple(location_vec + np.array([0, 1]))
    vicinity = [north, south, west, east]
    return vicinity

sqrt_2 = math.sqrt(2)
sqrt_8 = math.sqrt(8)
sqrt_18 = math.sqrt(18)
sqrt_32 = math.sqrt(32)

euclidean_distances_for_grid = {
    (3, 3): sqrt_2,
    (4, 4): sqrt_8,
    (5, 5): sqrt_18,
    (6, 6): sqrt_32
}


def parse_grid_observation(obs, num_agents, num_apples):

    agent_locations = []
    agent_levels = []
    for agent_id in range(num_agents):
        agent_location = obs[agent_id * 2], obs[(agent_id * 2) + 1]
        agent_level = obs[agent_id + ((num_agents + num_apples) * 2)]
        agent_locations.append(agent_location)
        agent_levels.append(int(agent_level))

    apple_locations = []
    apple_levels = []
    active_apples = []
    for apple_id in range(num_apples):
        apple_location = obs[(apple_id + num_agents) * 2], obs[((apple_id + num_agents) * 2) + 1]
        apple_level = obs[apple_id + ((num_agents + num_apples) * 2) + num_agents]
        active_apple = bool(obs[apple_id + ((num_agents + num_apples) * 2) + num_agents + num_apples])
        apple_locations.append(apple_location)
        apple_levels.append(int(apple_level))
        active_apples.append(bool(active_apple))

    return agent_locations, apple_locations, agent_levels, apple_levels, active_apples


def parse_partial_relative_observation(obs, num_agents, num_apples):

    num_teammates = num_agents - 1

    teammates_locations = []
    teammates_levels = []
    teammates_visible = []

    apples_location = []
    apples_level = []
    apples_active = []
    apples_visible = []

    teammates_locations_start_offset = 0
    apples_locations_start_offset = num_teammates * 2
    visibility_start_offset = apples_locations_start_offset + (num_apples * 2)
    levels_start_offset = visibility_start_offset + (num_teammates + num_apples)
    apples_active_start_offset = levels_start_offset + (num_teammates + num_apples)

    for teammate_id in range(num_teammates):

        teammate_is_visible = obs[visibility_start_offset + teammate_id]
        teammate_location = obs[teammates_locations_start_offset + (teammate_id * 2)], obs[teammates_locations_start_offset + (teammate_id * 2) + 1]
        teammate_level = obs[levels_start_offset + teammate_id]

        teammates_locations.append(teammate_location)
        teammates_visible.append(bool(teammate_is_visible))
        teammates_levels.append(teammate_level)

        if teammate_is_visible:
            print(f"Teammate #{teammate_id+1} visible at {teammate_location}, has level {teammate_level}")

    for apple_id in range(num_apples):

        apple_is_visible = obs[visibility_start_offset + num_teammates + apple_id]
        apple_location = obs[apples_locations_start_offset + (apple_id * 2)], obs[apples_locations_start_offset + (apple_id * 2) + 1]
        apple_level = obs[levels_start_offset + num_teammates + apple_id]
        apple_active = obs[apples_active_start_offset + apple_id]

        apples_location.append(apple_location)
        apples_level.append(apple_level)
        apples_active.append(apple_active)
        apples_visible.append(bool(apple_is_visible))
        if apple_is_visible:
            print(f"{'Apple' if apple_active else 'Foraged apple'} #{apple_id} at {apple_location}, {'has' if apple_active else 'had'} level {apple_level}")

    my_level = obs[-1]

    for t in range(num_teammates):
        teammate_location = teammates_locations[t]
        teammate_level = teammates_levels[t]
        print(f"Teammate {t+1}: {teammate_location} ({teammate_level})")

    return teammates_locations, teammates_levels, teammates_visible, apples_location, apples_level, apples_active, apples_visible, my_level


def find_closest_level_apple(agent_location, level, apple_locations, apple_levels, active_apples):

    distances = [
        cityblock(apple_locations[apple_id], agent_location) if active_apples[apple_id] and level >= apple_levels[apple_id] else math.inf
        for apple_id, apple_location in enumerate(apple_locations)
    ]
    return np.argmin(distances) if np.min(distances) != math.inf else None


def find_highest_level_apple(agent_location, apple_locations, apple_levels, active_apples):

    num_apples = len(apple_locations)

    apples_by_level = defaultdict(lambda: [])
    for apple_id in range(num_apples):
        if active_apples[apple_id]:
            apples_by_level[apple_levels[apple_id]].append(apple_id)

    if len(apples_by_level) > 0:
        max_level = max(apples_by_level)
        if len(apples_by_level[max_level]) == 1:
            return apples_by_level[max_level][0]
        elif len(apples_by_level[max_level]) > 1:
            apple_ids = apples_by_level[max_level]

            # This code sent the agent to the closest apple
            # Im changing so they go to the lowest id (for better coordination)
            """min_distance = math.inf
            apple_id = None
            for other_apple_id in apple_ids:
                distance = cityblock(apple_locations[other_apple_id], agent_location)
                if distance < min_distance:
                    apple_id = other_apple_id
                    min_distance = distance
            """
            apple_id = min(apple_ids)
            return apple_id
        else:
            return None
    else:
        return None


def greedy_direction(source, target):
    target_row, target_column = target
    source_row, source_column = source
    if target_row - source_row > 0: return SOUTH
    elif target_row - source_row < 0: return NORTH
    elif target_column - source_column > 0: return EAST
    elif target_column - source_column < 0: return WEST
    else: raise ValueError("Should be unreachable")


def pathfinding_distance(agent_id, target_location, agent_locations, apple_locations, active_apples, world_size):

    source_location = agent_locations[agent_id]

    obstacles = \
        [
            agent_locations[other_agent_id]
            for other_agent_id in range(len(agent_locations))
            if other_agent_id != agent_id
        ] + \
        [
            apple_locations[apple_id]
            for apple_id in range(len(apple_locations))
            if active_apples[apple_id] and apple_locations[apple_id] != target_location
        ]

    path, runs = pathfind(world_size, source_location, target_location, obstacles)
    return len(path)


def goto(agent_id, target_location, agent_locations, apple_locations, active_apples, world_size):

    source_location = agent_locations[agent_id]

    obstacles = \
        [
            agent_locations[other_agent_id]
            for other_agent_id in range(len(agent_locations))
            if other_agent_id != agent_id
        ] + \
        [
            apple_locations[apple_id]
            for apple_id in range(len(apple_locations))
            if active_apples[apple_id] and apple_locations[apple_id] != target_location
        ]

    path, runs = pathfind(world_size, source_location, target_location, obstacles)

    if len(path) > 0:

        next_position = path[1]
        diff = np.array(next_position) - np.array((source_location[1], source_location[0]))

        if diff[0] >= 1: action_meaning = "right"
        elif diff[0] <= -1: action_meaning = "left"
        elif diff[1] >= 1: action_meaning = "down"
        elif diff[1] <= -1: action_meaning = "up"
        else: action_meaning = "noop"

    else:
        action_meaning = "noop"

    # print(grid.grid_str(path=path, start=start, end=end))

    action = ACTION_MEANINGS.index(action_meaning)
    return action


def pathfind(world_size, source, target, obstacles):

    matrix = np.ones(world_size)
    for obstacle in obstacles:
        matrix[obstacle] = 0

    grid = Grid(matrix=matrix)
    start = grid.node(source[1], source[0])
    end = grid.node(target[1], target[0])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)

    return path, runs


class AStarNode(object):
    def __init__(self, position, parent, cost, heuristic):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __hash__(self):
        return self.position.__hash__()

    def __eq__(self, other):
        return self.position == other.position


def a_star_search(source, obstacles, target):

    if source == target:
        return (0, 0), 0

    obstacles = obstacles - {target}

    def heuristic(position):
        def distance(source, target):
            dx = min((source[0] - target[0]), (target[0] - source[0]))
            dy = min((source[1] - target[1]), (target[1] - source[1]))
            return dx, dy
        return sum(distance(source, position))

    # each item in the queue contains (heuristic+cost, cost, position, parent)
    initial_node = AStarNode(source, None, 0, heuristic(source))
    queue = [
        AStarNode(n, initial_node, 1, heuristic(n))
        for n in compute_vicinity(source) if n not in obstacles
    ]

    heapify(queue)
    visited = set()
    visited.add(source)
    current = initial_node

    while len(queue) > 0:

        current = heappop(queue)

        if current.position in visited:
            continue

        visited.add(current.position)

        if current.position == target:
            break

        vicinity = compute_vicinity(current.position)
        for position in vicinity:
            if position not in obstacles:
                new_node = AStarNode(position, current, current.cost + 1, heuristic(position))
                heappush(queue, new_node)

    if target not in visited:
        return None

    i = 1
    while current.parent != initial_node:
        current = current.parent
        i += 1

    return greedy_direction(source, current.position)
