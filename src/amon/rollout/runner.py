"""Runs one episode of a frozen policy and records it."""

from dataclasses import dataclass

import cv2
import numpy as np

from amon.policy.act import ActAdapter
from amon.sim.env import AlohaSim
from amon.sim.perturb import Perturbation
from amon.sim.state import SimState, StateReader

FRAME_RESIZE = "cv2.INTER_AREA, aspect ratio not kept"
VIDEO_CAMERA = "angle"  # second view in review videos; a drop is hard to see from above


@dataclass(frozen=True)
class EpisodeTrace:
    """Per-step record of one episode of T steps.

    Every array except `reward` describes frame t, before action t is applied. `reward[t]` is the
    outcome of action t.
    """

    seed: int
    success: bool
    reward: np.ndarray  # [T] int8
    frames: np.ndarray  # [T, h, w, 3] uint8, top camera
    agent_pos: np.ndarray  # [T, 14] float32
    action_exec: np.ndarray  # [T, 14] float32
    obj_pos: np.ndarray  # [T, K, 3] float32
    contact: np.ndarray  # [T, K, 2] bool, arms ordered left, right
    held: np.ndarray  # [T, K, 2] bool
    chunks: np.ndarray | None  # [T, H, 14] float32 when shadow queries are on
    video: np.ndarray | None  # [T, 480, 1280, 3] uint8 when requested
    perturb_step: int  # first step whose physics were perturbed, -1 if none
    perturb_parameters: dict[str, float]


def run_episode(
    sim: AlohaSim,
    policy: ActAdapter,
    reader: StateReader,
    seed: int,
    frame_size: tuple[int, int],
    shadow_queries: bool,
    perturbation: Perturbation | None = None,
    record_video: bool = False,
) -> EpisodeTrace:
    """Rolls out one seeded episode until success or the step limit.

    Args:
        sim: Environment to run in.
        policy: Frozen policy; its executor is reset here.
        reader: Reads the privileged state for labels and perturbation timing.
        seed: Episode seed, which fixes the initial object pose.
        frame_size: Stored frame height and width.
        shadow_queries: Whether to also log a full chunk at every step.
        perturbation: Fresh perturbation for this episode, if any.
        record_video: Whether to keep native-resolution frames from two cameras.

    Returns:
        The recorded episode.
    """
    policy.reset()
    observation = sim.reset(seed)
    rewards: list[int] = []
    frames: list[np.ndarray] = []
    agent_positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    states: list[SimState] = []
    chunks: list[np.ndarray] = []
    video: list[np.ndarray] = []
    success = False
    done = False
    phase = 0
    try:
        while not done:
            state = reader.read(sim.physics)
            states.append(state)
            frames.append(
                cv2.resize(observation.image, frame_size[::-1], interpolation=cv2.INTER_AREA)
            )
            agent_positions.append(observation.agent_pos)
            if record_video:
                video.append(np.hstack([observation.image, sim.render(VIDEO_CAMERA)]))
            if shadow_queries:
                chunks.append(policy.shadow_chunk(observation))
            action = policy.act(observation)
            actions.append(action)
            if perturbation is not None:
                perturbation.before_step(sim.physics, len(rewards), state, phase)
            hook = None
            if perturbation is not None and perturbation.substep_hook_needed():
                hook = perturbation.substep
            result = sim.step(action, hook)
            rewards.append(result.reward)
            phase = max(phase, result.reward)
            success = success or result.success
            done = result.done
            observation = result.observation
    finally:
        # Model edits such as friction would otherwise leak into the next episode.
        if perturbation is not None:
            perturbation.restore(sim.physics)
    return EpisodeTrace(
        seed=seed,
        success=success,
        reward=np.array(rewards, dtype=np.int8),
        frames=np.array(frames, dtype=np.uint8),
        agent_pos=np.array(agent_positions, dtype=np.float32),
        action_exec=np.array(actions, dtype=np.float32),
        obj_pos=np.array([state.obj_pos for state in states], dtype=np.float32),
        contact=np.array([state.contact for state in states], dtype=bool),
        held=np.array([state.held for state in states], dtype=bool),
        chunks=np.array(chunks, dtype=np.float32) if shadow_queries else None,
        video=np.array(video, dtype=np.uint8) if record_video else None,
        perturb_step=-1 if perturbation is None else perturbation.step_fired,
        perturb_parameters={} if perturbation is None else perturbation.parameters,
    )
