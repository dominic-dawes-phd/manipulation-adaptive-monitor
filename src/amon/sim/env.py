"""gym-aloha wrapper: one task, one environment per process.

gym-aloha keeps the object start pose in a module-level variable (`BOX_POSE`), so two live
environments in one process would overwrite each other's reset state.
"""

import warnings
from dataclasses import dataclass

import gym_aloha  # noqa: F401  (registers the environments)
import gymnasium as gym
import numpy as np
from dm_control import mujoco

from amon.settings import TaskSettings


@dataclass(frozen=True)
class Observation:
    """What the policy sees at one control step."""

    image: np.ndarray  # [480, 640, 3] uint8, top camera
    agent_pos: np.ndarray  # [14] float64, joint state


@dataclass(frozen=True)
class StepResult:
    """Outcome of one control step."""

    observation: Observation
    reward: int
    success: bool
    done: bool


class AlohaSim:
    """A seeded gym-aloha environment with access to the physics state."""

    def __init__(self, task: TaskSettings) -> None:
        """Creates the environment with the checkpoint's own evaluation settings."""
        self._env = gym.make(
            task.env_id,
            obs_type=task.obs_type,
            render_mode="rgb_array",
            max_episode_steps=task.episode_len,
        )

    @property
    def physics(self) -> mujoco.Physics:
        """The dm_control physics, for labels and perturbations."""
        return self._env.unwrapped._env.physics

    def reset(self, seed: int) -> Observation:
        """Starts an episode whose initial object pose is fixed by `seed`."""
        raw, _ = self._env.reset(seed=seed)
        return _to_observation(raw)

    def step(self, action: np.ndarray) -> StepResult:
        """Applies one 14-D joint-space action for one control step."""
        with warnings.catch_warnings():
            # gymnasium's checker objects to joint values outside gym-aloha's nominal boxes.
            warnings.simplefilter("ignore", UserWarning)
            raw, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            observation=_to_observation(raw),
            reward=int(reward),
            success=bool(info["is_success"]),
            done=bool(terminated or truncated),
        )

    def close(self) -> None:
        """Releases the environment."""
        self._env.close()


def _to_observation(raw: dict) -> Observation:
    """Copies the arrays out of a raw gym-aloha observation."""
    return Observation(image=raw["pixels"]["top"], agent_pos=raw["agent_pos"].copy())
