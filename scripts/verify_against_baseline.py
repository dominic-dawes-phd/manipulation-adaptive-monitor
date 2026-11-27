"""Checks that the current pipeline reproduces recorded baseline episodes bit for bit.

Only meaningful on the machine that recorded them (rendering is hardware dependent).

Usage: python scripts/verify_against_baseline.py <baseline_run_dir> [--episodes N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from amon.policy.act import ActAdapter, enable_determinism
from amon.rollout.runner import run_episode
from amon.settings import load_settings
from amon.sim.env import AlohaSim
from amon.sim.state import StateReader


def main() -> int:
    """Re-runs the first episodes of each task and compares every recorded array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    arguments = parser.parse_args()
    enable_determinism()
    settings = load_settings()
    mismatches = 0
    for name, task in settings.tasks.items():
        sim = AlohaSim(task, settings.render)
        policy = ActAdapter(task)
        reader = StateReader(sim.physics, task, settings)
        paths = sorted((arguments.baseline / "baseline" / name).glob("ep_*.npz"))
        start = time.monotonic()
        for path in paths[: arguments.episodes]:
            with np.load(path) as recorded:
                trace = run_episode(
                    sim,
                    policy,
                    reader,
                    int(recorded["seed"]),
                    settings.render.store_size,
                    shadow_queries=True,
                )
                same = (
                    np.array_equal(trace.action_exec, recorded["action_exec"])
                    and np.array_equal(trace.agent_pos, recorded["agent_pos"])
                    and np.array_equal(trace.reward, recorded["reward"])
                )
            mismatches += not same
            print(f"{name} {path.stem}: {'identical' if same else 'DIFFERENT'}")
        print(f"{name}: {(time.monotonic() - start) / arguments.episodes:.1f} s/episode")
        sim.close()
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
