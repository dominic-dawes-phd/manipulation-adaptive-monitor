"""Runs one episode of a frozen policy."""

from dataclasses import dataclass

import numpy as np

from amon.policy.act import ActAdapter
from amon.sim.env import AlohaSim


@dataclass(frozen=True)
class EpisodeTrace:
    """Per-step record of one episode of T steps."""

    seed: int
    success: bool
    reward: np.ndarray  # [T] int8
    agent_pos: np.ndarray  # [T, 14] float32
    action_exec: np.ndarray  # [T, 14] float32
    chunks: np.ndarray | None  # [T, H, 14] float32 when shadow queries are on


def run_episode(sim: AlohaSim, policy: ActAdapter, seed: int, shadow_queries: bool) -> EpisodeTrace:
    """Rolls out one seeded episode until success or the step limit.

    Args:
        sim: Environment to run in.
        policy: Frozen policy; its executor is reset here.
        seed: Episode seed, which fixes the initial object pose.
        shadow_queries: Whether to also log a full chunk at every step.

    Returns:
        The recorded episode.
    """
    policy.reset()
    observation = sim.reset(seed)
    rewards: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    success = False
    done = False
    while not done:
        if shadow_queries:
            chunks.append(policy.shadow_chunk(observation))
        action = policy.act(observation)
        states.append(observation.agent_pos)
        actions.append(action)
        result = sim.step(action)
        rewards.append(result.reward)
        success = success or result.success
        done = result.done
        observation = result.observation
    return EpisodeTrace(
        seed=seed,
        success=success,
        reward=np.array(rewards, dtype=np.int8),
        agent_pos=np.array(states, dtype=np.float32),
        action_exec=np.array(actions, dtype=np.float32),
        chunks=np.array(chunks, dtype=np.float32) if shadow_queries else None,
    )
