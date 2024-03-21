import time
from typing import Sequence
from gym import Env
from environments import LevelBasedForaging
from environments.level_based_foraging.agents import MaxLevelForager


def run_episode(env: Env, agents: Sequence, render=True, render_sleep=0.05):
    observations = env.reset()
    if render:
        env.render()
        time.sleep(render_sleep)
    terminal = False
    while not terminal:
        actions = [
            agents[agent_id].act(observations[agent_id])
            for agent_id in range(len(agents))
        ]
        observations, rewards, terminals, info = env.step(actions)
        if render:
            env.render()
            time.sleep(render_sleep)

        terminal = any(terminals)
        if terminal:
            observations = env.reset()
            if render: env.render()


if __name__ == '__main__':

    env = LevelBasedForaging(
        num_agents=4,
        max_level=1,
        world_size=(10, 10),
        max_apples=2,
        always_max_apples=True
    )

    agents = [
        MaxLevelForager(agent_id, env.num_agents, env.num_apples, env.world_size)
        for agent_id in range(env.num_agents)
    ]

    for ep in range(32):
        run_episode(env, agents)
