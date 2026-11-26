"""Runs one shard of one stage: `python -m experiments.run --spec ... --stage ...`."""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from amon.policy.act import enable_determinism
from amon.rollout.baseline import report_baseline, run_baseline
from amon.rollout.manifest import write_manifest
from amon.rollout.sweep import expand_cells, run_rollouts
from amon.settings import REPO_ROOT, Settings, load_settings
from amon.version import AMON_VERSION

LOGGER = logging.getLogger("experiments.run")
STAGES = ("baseline", "baseline_report", "rollout")


def main() -> int:
    """Parses arguments and runs the requested stage.

    Returns:
        0 if every episode succeeded in running, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--name", help="run name; defaults to the name in the spec")
    parser.add_argument("--run-dir", type=Path, help="existing run folder to continue")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, help="run only the first N episodes (plumbing checks, never results)"
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # dm_control installs a root handler on import
    )

    spec = yaml.safe_load(arguments.spec.read_text(encoding="utf-8"))
    settings = load_settings(overrides=spec.get("overrides"))
    run_dir = _prepare_run_dir(arguments.run_dir, arguments.name or spec["name"], arguments.spec)
    enable_determinism()
    write_manifest(run_dir, spec, settings)
    LOGGER.info(
        "amon %s, %s stage %s, shard %d/%d, %s",
        AMON_VERSION,
        spec["name"],
        arguments.stage,
        arguments.shard,
        arguments.of,
        run_dir,
    )

    stages = spec["stages"] if arguments.stage == "all" else [arguments.stage]
    failed = 0
    for stage in stages:
        if stage == "baseline":
            failed += _run_baseline_stage(
                settings, spec, run_dir, arguments.shard, arguments.of, arguments.limit
            )
        elif stage == "rollout":
            keys = expand_cells(spec, settings)[: arguments.limit][arguments.shard :: arguments.of]
            video = spec.get("store_video", settings.render.store_video)
            failed += len(run_rollouts(keys, run_dir, settings, video))
        elif stage == "baseline_report":
            summary = report_baseline(settings, spec["tasks"], run_dir)
            (run_dir / "eval" / "baseline_summary.txt").write_text(summary + "\n", encoding="utf-8")
            print(summary)
        else:
            raise ValueError(f"stage {stage!r} is not implemented yet (see HANDOFF.md section 5)")
    return 1 if failed else 0


def _run_baseline_stage(
    settings: Settings, spec: dict, run_dir: Path, shard: int, of: int, limit: int | None
) -> int:
    """Runs this shard's nominal episodes for every task and returns the failure count."""
    start, end = settings.splits[spec["split"]]
    seeds = list(range(start, end))[:limit][shard::of]
    failed = 0
    for name in spec["tasks"]:
        failed += len(
            run_baseline(
                settings.tasks[name],
                seeds,
                run_dir / "baseline" / name,
                settings,
                shadow_queries=True,
            )
        )
    return failed


def _prepare_run_dir(existing: Path | None, name: str, spec_path: Path) -> Path:
    """Returns the run folder, creating a stamped one when none was given."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = existing or REPO_ROOT / "runs" / f"{name}_{stamp}"
    for child in ("episodes", "eval", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    if not (run_dir / "spec.yaml").exists():
        shutil.copyfile(spec_path, run_dir / "spec.yaml")
    return run_dir


if __name__ == "__main__":
    sys.exit(main())
