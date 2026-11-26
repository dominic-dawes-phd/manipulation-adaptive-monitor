"""gym-aloha wrapper: one task, one environment per process.

gym-aloha keeps the object start pose in a module-level variable (`BOX_POSE`), so two live
environments in one process would overwrite each other's reset state.
"""

import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import gym_aloha  # noqa: F401  (registers the environments)
import gymnasium as gym
import numpy as np
from dm_control import mujoco
from gym_aloha.tasks.sim import BimanualViperXTask

from amon.settings import TaskSettings

SubstepHook = Callable[[mujoco.Physics], None]
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640  # fixed by the checkpoints' input features


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
        # gym-aloha renders three cameras per step and returns one.
        self._env.unwrapped._env.task.get_observation = _top_camera_observation

    @property
    def physics(self) -> mujoco.Physics:
        """The dm_control physics, for labels and perturbations."""
        return self._env.unwrapped._env.physics

    def reset(self, seed: int) -> Observation:
        """Starts an episode whose initial object pose is fixed by `seed`."""
        raw, _ = self._env.reset(seed=seed)
        return _to_observation(raw)

    def step(self, action: np.ndarray, substep_hook: SubstepHook | None = None) -> StepResult:
        """Applies one 14-D joint-space action for one control step.

        Args:
            action: Joint targets.
            substep_hook: Called before every physics substep, for perturbations that must react
                faster than the control rate. None leaves gym-aloha's own stepping untouched.
        """
        if substep_hook is not None:
            self.physics.step = partial(_step_with_hook, self.physics, substep_hook)
        try:
            with warnings.catch_warnings():
                # gymnasium's checker objects to joint values outside gym-aloha's nominal boxes.
                warnings.simplefilter("ignore", UserWarning)
                raw, reward, terminated, truncated, info = self._env.step(action)
        finally:
            if substep_hook is not None:
                del self.physics.step
        return StepResult(
            observation=_to_observation(raw),
            reward=int(reward),
            success=bool(info["is_success"]),
            done=bool(terminated or truncated),
        )

    def render(self, camera: str) -> np.ndarray:
        """Renders another camera, for review videos only."""
        return self.physics.render(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, camera_id=camera)

    def step_without_rendering(
        self, action: np.ndarray, substep_hook: SubstepHook | None = None
    ) -> int:
        """Advances the physics like `step`, with no image or step limit; returns the reward.

        Only for replaying recorded actions, for example to calibrate perturbations.
        """
        dm_env = self._env.unwrapped._env
        dm_env.task.before_step(action, dm_env.physics)
        if substep_hook is None:
            dm_env.physics.step(dm_env._n_sub_steps)
        else:
            _step_with_hook(dm_env.physics, substep_hook, dm_env._n_sub_steps)
        return int(dm_env.task.get_reward(dm_env.physics))

    def close(self) -> None:
        """Releases the environment."""
        self._env.close()


def _to_observation(raw: dict) -> Observation:
    """Copies the arrays out of a raw gym-aloha observation."""
    return Observation(image=raw["pixels"]["top"], agent_pos=raw["agent_pos"].copy())


def _top_camera_observation(physics: mujoco.Physics) -> OrderedDict:
    """Builds the part of gym-aloha's observation that the environment returns."""
    image = physics.render(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, camera_id="top")
    return OrderedDict(qpos=BimanualViperXTask.get_qpos(physics), images={"top": image})


def _step_with_hook(physics: mujoco.Physics, hook: SubstepHook, nstep: int = 1) -> None:
    """Runs `nstep` physics substeps, calling the hook before each."""
    for _ in range(nstep):
        hook(physics)
        type(physics).step(physics, 1)
