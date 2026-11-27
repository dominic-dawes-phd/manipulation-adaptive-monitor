"""Checks that two runs of the same episodes are bit-identical (server determinism, D-12, D-22).

Compares every array of every data.npz (values and dtype) and every meta.json. Exits 1 on any
difference or on an episode present in only one run.

    python scripts/compare_runs.py runs/server_check_<stamp>/gpu0 runs/server_check_<stamp>/gpu1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    """Compares the runs and returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[compare] %(message)s", force=True)

    first_ids = {
        path.parent.relative_to(arguments.first)
        for path in arguments.first.glob("episodes/*/*/*/ep_*/meta.json")
    }
    second_ids = {
        path.parent.relative_to(arguments.second)
        for path in arguments.second.glob("episodes/*/*/*/ep_*/meta.json")
    }
    differences = [f"only in one run: {episode}" for episode in sorted(first_ids ^ second_ids)]
    for episode in sorted(first_ids & second_ids):
        first_meta = json.loads((arguments.first / episode / "meta.json").read_text())
        second_meta = json.loads((arguments.second / episode / "meta.json").read_text())
        if first_meta != second_meta:
            differences.append(f"{episode}: meta.json differs")
        with (
            np.load(arguments.first / episode / "data.npz") as first,
            np.load(arguments.second / episode / "data.npz") as second,
        ):
            if set(first.files) != set(second.files):
                differences.append(f"{episode}: different arrays")
                continue
            for name in first.files:
                if first[name].dtype != second[name].dtype or not np.array_equal(
                    first[name], second[name]
                ):
                    differences.append(f"{episode}: {name} differs")
    for line in differences:
        logging.error(line)
    compared = len(first_ids & second_ids)
    if differences or not compared:
        logging.error("FAIL: %d differences over %d episodes", len(differences), compared)
        return 1
    logging.info("PASS: %d episodes bit-identical", compared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
