"""M1 baseline: nominal success rates of the frozen checkpoints.

Traces are small arrays only; the full episode format (spec section 2) arrives in M2.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from amon.eval.stats import exact_mcnemar, wilson_interval
from amon.policy.act import ActAdapter
from amon.rollout.runner import run_episode
from amon.settings import Settings, TaskSettings
from amon.sim.env import AlohaSim

LOGGER = logging.getLogger(__name__)
PROGRESS_INTERVAL = 10


def run_baseline(
    task: TaskSettings, seeds: list[int], out_dir: Path, shadow_queries: bool
) -> list[int]:
    """Rolls out every seed that has no saved trace yet.

    Args:
        task: Task to run.
        seeds: Episode seeds of this shard.
        out_dir: Where `ep_<seed>.npz` files go.
        shadow_queries: Whether to query a chunk at every step.

    Returns:
        Seeds whose episode raised an error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [seed for seed in seeds if not _trace_path(out_dir, seed).exists()]
    sim = AlohaSim(task)
    policy = ActAdapter(task)
    failed: list[int] = []
    start = time.monotonic()
    for index, seed in enumerate(pending, start=1):
        try:
            trace = run_episode(sim, policy, seed, shadow_queries)
        except Exception:  # A broken episode must not stop the shard.
            LOGGER.exception("EPISODE_FAILED %s/none/baseline/ep_%06d", task.name, seed)
            failed.append(seed)
            continue
        # Write then rename, so that an interrupted run never leaves a truncated trace.
        partial = out_dir / f"partial_{seed:06d}.npz"
        np.savez_compressed(
            partial,
            seed=trace.seed,
            success=trace.success,
            reward=trace.reward,
            agent_pos=trace.agent_pos,
            action_exec=trace.action_exec,
        )
        partial.rename(_trace_path(out_dir, seed))
        if index % PROGRESS_INTERVAL == 0 or index == len(pending):
            seconds = (time.monotonic() - start) / index
            remaining = seconds * (len(pending) - index) / 60
            LOGGER.info(
                "%s %d/%d, %.1f s/episode, %.0f min left",
                task.name,
                index,
                len(pending),
                seconds,
                remaining,
            )
    sim.close()
    return failed


def report_baseline(settings: Settings, task_names: list[str], run_dir: Path) -> str:
    """Summarizes saved traces into eval/baseline.csv and a printable text block."""
    rows: list[dict[str, str | int | float]] = []
    lines: list[str] = []
    for name in task_names:
        task = settings.tasks[name]
        episodes = _load_episodes(run_dir / "baseline" / name)
        result = wilson_interval(int(episodes["success"].sum()), len(episodes), settings.stats.ci)
        rows.append(
            {
                "task": name,
                "successes": result.successes,
                "episodes": result.trials,
                "rate": result.estimate,
                "wilson_low": result.low,
                "wilson_high": result.high,
                "model_card_rate": task.published_success_rate,
            }
        )
        lines.append(
            f"{name}: {result.successes}/{result.trials} = {result.estimate:.1%} "
            f"[{result.low:.1%}, {result.high:.1%}] Wilson {settings.stats.ci:.0%}; "
            f"model card {task.published_success_rate:.1%}"
        )
        for highest, count in episodes["max_reward"].value_counts().sort_index().items():
            lines.append(f"  highest reward stage {highest}: {count} episodes")
        if task.published_per_episode:
            lines.append(_compare_with_published(task, episodes))
        episodes.to_csv(run_dir / "eval" / f"baseline_{name}.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "eval" / "baseline.csv", index=False)
    return "\n".join(lines)


def _compare_with_published(task: TaskSettings, episodes: pd.DataFrame) -> str:
    """Compares with the checkpoint's published outcomes on the same seeds."""
    path = hf_hub_download(task.policy_repo, "eval_info.json", revision=task.policy_revision)
    records = json.loads(Path(path).read_text(encoding="utf-8"))["per_episode"]
    published = {record["seed"]: record["success"] for record in records}
    theirs = episodes["seed"].map(published).to_numpy(dtype=bool)
    ours = episodes["success"].to_numpy(dtype=bool)
    paired = exact_mcnemar(ours, theirs)
    return (
        f"  published on the same seeds: {theirs.sum()}/{len(theirs)} = {theirs.mean():.1%}; "
        f"only ours {paired.only_first}, only published {paired.only_second}, "
        f"exact McNemar p = {paired.p_value:.3f}"
    )


def _load_episodes(task_dir: Path) -> pd.DataFrame:
    """Reads one row per saved trace."""
    rows = []
    for path in sorted(task_dir.glob("ep_*.npz")):
        with np.load(path) as trace:
            rows.append(
                {
                    "seed": int(trace["seed"]),
                    "success": bool(trace["success"]),
                    "steps": len(trace["reward"]),
                    "max_reward": int(trace["reward"].max()),
                    "sum_reward": int(trace["reward"].sum()),
                }
            )
    if not rows:
        raise FileNotFoundError(f"no baseline traces in {task_dir}")
    return pd.DataFrame(rows)


def _trace_path(out_dir: Path, seed: int) -> Path:
    """Returns the trace file of one seed."""
    return out_dir / f"ep_{seed:06d}.npz"
