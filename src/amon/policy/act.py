"""LeRobot ACT adapter: the executor and the logging-only shadow query.

Verified against LeRobot 0.3.3: `select_action` owns the action queue (or temporal ensembler),
and `predict_action_chunk` is a pure forward pass that touches neither.
"""

import os

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy

from amon.settings import TaskSettings
from amon.sim.env import Observation


def enable_determinism() -> None:
    """Makes torch raise on nondeterministic operations. Call before any CUDA work."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class ActAdapter:
    """A frozen ACT checkpoint executed exactly as its own config.json prescribes."""

    def __init__(self, task: TaskSettings, device: str = "cuda") -> None:
        """Loads the pinned checkpoint revision."""
        self._device = torch.device(device)
        self._policy = ACTPolicy.from_pretrained(
            task.policy_repo, revision=task.policy_revision, strict=True
        )
        self._policy.to(self._device).eval()
        config = self._policy.config
        self.chunk_size: int = config.chunk_size
        self.n_action_steps: int = config.n_action_steps
        self.temporal_ensemble_coeff: float | None = config.temporal_ensemble_coeff

    def reset(self) -> None:
        """Clears the executor's queue at the start of an episode."""
        self._policy.reset()

    def act(self, observation: Observation) -> np.ndarray:
        """Returns the action to execute, shape [14], float32."""
        action = self._policy.select_action(self._to_batch(observation))
        return action.squeeze(0).cpu().numpy()

    def shadow_chunk(self, observation: Observation) -> np.ndarray:
        """Returns a full predicted chunk, shape [H, 14], float32, without affecting execution."""
        chunk = self._policy.predict_action_chunk(self._to_batch(observation))
        return chunk.squeeze(0).cpu().numpy()

    def _to_batch(self, observation: Observation) -> dict[str, torch.Tensor]:
        """Converts an observation with LeRobot's own evaluation preprocessing."""
        batch = preprocess_observation(
            {"pixels": {"top": observation.image}, "agent_pos": observation.agent_pos}
        )
        return {key: value.to(self._device) for key, value in batch.items()}
