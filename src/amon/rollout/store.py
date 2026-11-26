"""Episode files on disk (spec section 2)."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from amon.rollout.runner import EpisodeTrace
from amon.settings import Settings
from amon.sim.labels import boundary_steps, label_failure, phase_from_reward
from amon.sim.state import LEFT, RIGHT
from amon.version import AMON_VERSION


@dataclass(frozen=True)
class EpisodeKey:
    """Identifies one episode of a run."""

    task: str
    perturbation: str
    split: str
    seed: int

    @property
    def episode_id(self) -> str:
        """Relative folder of the episode, also used as its id."""
        return f"{self.task}/{self.perturbation}/{self.split}/ep_{self.seed:06d}"


def episode_dir(run_dir: Path, key: EpisodeKey) -> Path:
    """Returns the folder of one episode."""
    return run_dir / "episodes" / key.episode_id


def save_episode(
    run_dir: Path, key: EpisodeKey, trace: EpisodeTrace, settings: Settings
) -> dict[str, str | int | bool | list[int] | dict[str, float]]:
    """Labels an episode and writes data.npz, meta.json and optionally video.mp4.

    Args:
        run_dir: Run folder.
        key: Episode identity.
        trace: Recorded episode; its shadow chunks must be present.
        settings: Global settings.

    Returns:
        The written metadata.
    """
    if trace.chunks is None:
        raise ValueError("stored episodes need shadow queries")
    phase = phase_from_reward(trace.reward)
    failure = label_failure(
        key.task,
        trace.success,
        phase,
        trace.held,
        trace.contact,
        trace.action_exec,
        settings.labels,
        settings.robot,
        key.perturbation,
        trace.perturb_step,
    )
    meta = {
        "episode_id": key.episode_id,
        "task": key.task,
        "perturbation": key.perturbation,
        "perturb_parameters": trace.perturb_parameters,
        "perturb_step": trace.perturb_step,
        "success": trace.success,
        "onset_step": failure.onset,
        "failure_type": failure.kind,
        "boundary_steps": boundary_steps(phase).tolist(),
        "seed": key.seed,
        "split": key.split,
        "sim_version": AMON_VERSION,
    }
    target = episode_dir(run_dir, key)
    # Write to a sibling and rename, so that an interrupted run leaves no half-written episode.
    partial = target.with_name(f"partial_{target.name}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.savez_compressed(
        partial / "data.npz",
        frames=trace.frames,
        agent_pos=trace.agent_pos,
        action_exec=trace.action_exec,
        chunks=trace.chunks.astype(np.float16),
        reward=trace.reward,
        phase=phase,
        contact_left=trace.contact[:, :, LEFT].any(axis=1),
        contact_right=trace.contact[:, :, RIGHT].any(axis=1),
        contact=trace.contact,
        held=trace.held,
        obj_pos=trace.obj_pos,
    )
    if trace.video is not None:
        imageio.mimwrite(
            partial / "video.mp4",
            list(trace.video),
            fps=settings.control_hz,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
    (partial / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    partial.rename(target)
    return meta
