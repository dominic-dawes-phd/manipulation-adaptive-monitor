"""The rollout stage: expands a sweep spec into episodes and runs one shard of them."""

import logging
import time
from pathlib import Path

from amon.policy.act import ActAdapter
from amon.rollout.runner import run_episode
from amon.rollout.store import EpisodeKey, episode_dir, save_episode
from amon.settings import Settings
from amon.sim.env import AlohaSim
from amon.sim.perturb import PERTURBATIONS, make_perturbation
from amon.sim.state import StateReader

LOGGER = logging.getLogger(__name__)
PROGRESS_INTERVAL = 10
SEEDS_PER_PERTURBATION = 1000


def expand_cells(spec: dict, settings: Settings) -> list[EpisodeKey]:
    """Lists every episode of a sweep spec.

    Args:
        spec: Sweep spec with `tasks`, `cells` and optionally `only` (episode ids to rerun).
        settings: Global settings.

    Returns:
        Episode keys, grouped by task.

    Raises:
        ValueError: If a cell asks for more seeds than its range holds.
    """
    keys: list[EpisodeKey] = []
    for task in spec["tasks"]:
        for cell in spec["cells"]:
            if task not in cell.get("tasks", spec["tasks"]):
                continue
            start, end = _seed_range(cell["perturbation"], cell["split"], settings)
            if start + cell["episodes"] > end:
                raise ValueError(f"cell {cell} needs more seeds than [{start}, {end})")
            for seed in range(start, start + cell["episodes"]):
                keys.append(EpisodeKey(task, cell["perturbation"], cell["split"], seed))
    if "only" in spec:
        wanted = set(spec["only"])
        keys = [key for key in keys if key.episode_id in wanted]
    return keys


def run_rollouts(
    keys: list[EpisodeKey], run_dir: Path, settings: Settings, record_video: bool
) -> list[EpisodeKey]:
    """Rolls out and stores every episode that is not on disk yet.

    Args:
        keys: This shard's episodes.
        run_dir: Run folder.
        settings: Global settings.
        record_video: Whether to also write native-resolution review videos.

    Returns:
        Episodes that raised an error.
    """
    failed: list[EpisodeKey] = []
    for task_name in dict.fromkeys(key.task for key in keys):
        task = settings.tasks[task_name]
        pending = [
            key
            for key in keys
            if key.task == task_name and not (episode_dir(run_dir, key) / "meta.json").exists()
        ]
        if not pending:
            continue
        sim = AlohaSim(task, settings.render)
        policy = ActAdapter(task)
        reader = StateReader(sim.physics, task, settings)
        start = time.monotonic()
        for index, key in enumerate(pending, start=1):
            try:
                trace = run_episode(
                    sim,
                    policy,
                    reader,
                    key.seed,
                    settings.render.store_size,
                    shadow_queries=True,
                    perturbation=make_perturbation(key.perturbation, task, settings, key.seed),
                    record_video=record_video,
                )
                save_episode(run_dir, key, trace, settings)
            except Exception:  # A broken episode must not stop the shard.
                LOGGER.exception("EPISODE_FAILED %s", key.episode_id)
                failed.append(key)
                continue
            if index % PROGRESS_INTERVAL == 0 or index == len(pending):
                seconds = (time.monotonic() - start) / index
                LOGGER.info(
                    "%s %d/%d, %.1f s/episode, %.0f min left",
                    task_name,
                    index,
                    len(pending),
                    seconds,
                    seconds * (len(pending) - index) / 60,
                )
        sim.close()
    return failed


def _seed_range(perturbation: str, split: str, settings: Settings) -> tuple[int, int]:
    """Returns the half-open seed range of a cell; perturbed test episodes have their own."""
    if perturbation != "none" and split == "test":
        index = PERTURBATIONS.index(perturbation)
        start = settings.perturbed_test_start + SEEDS_PER_PERTURBATION * index
        return start, start + SEEDS_PER_PERTURBATION
    if perturbation != "none" and split != "dev":
        raise ValueError(f"perturbed episodes belong to test or dev, not {split}")
    return settings.splits[split]
