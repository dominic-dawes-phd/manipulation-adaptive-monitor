"""Does chunk disagreement rise after a perturbation? A dev-split check before M3 (plan step 1).

Computes d(t) (spec section 8) on the perturbed dev review episodes and on their unperturbed
counterparts (same seeds, identical up to the perturbation step). Sigma comes from the M1
baseline successes and the percentile reference from all steps of the unperturbed episodes;
both stand in for calib (provisional; dev only, never reported). For every perturbed episode,
the WINDOW steps after the perturbation are compared with the same steps of its counterpart.

    python scripts/trigger_signal_check.py --review runs/review_<stamp> \
        --nominal runs/review_nominal_<stamp> --baseline runs/baseline_<stamp>
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np

from amon.rollout.manifest import write_manifest
from amon.settings import REPO_ROOT, Settings, load_settings
from amon.trigger.features import gripper_flips, group_disagreement, normalization_sigma

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)

WINDOW = 20
LEVELS = (0.95, 0.99)
PLOT_RANGE = (-30, 60)
Episode = dict[str, np.ndarray | int | str | bool]


def main() -> None:
    """Writes the aligned curves, the per-episode table and a logged summary to a new run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    settings = load_settings()
    run_dir = REPO_ROOT / "runs" / f"trigger_check_{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "eval").mkdir()
    spec = {
        "name": "trigger_check",
        "review": str(arguments.review),
        "nominal": str(arguments.nominal),
        "baseline": str(arguments.baseline),
        "window": WINDOW,
    }
    write_manifest(run_dir, spec, settings)

    rows = []
    for task in settings.tasks:
        sigma = baseline_sigma(arguments.baseline, task)
        nominal = {ep["seed"]: ep for ep in load_episodes(arguments.nominal, task, sigma, settings)}
        reference = np.sort(np.concatenate([ep["d"][1:] for ep in nominal.values()]))
        perturbed = load_episodes(arguments.review, task, sigma, settings)
        for ep in [*nominal.values(), *perturbed]:
            ep["percentile"] = np.searchsorted(reference, ep["d"], side="right") / len(reference)
        for perturbation in sorted({str(ep["perturbation"]) for ep in perturbed}):
            group = [ep for ep in perturbed if ep["perturbation"] == perturbation]
            fired = [ep for ep in group if int(ep["perturb"]) >= 0]
            logging.info("%s/%s: %d of %d fired", task, perturbation, len(fired), len(group))
            group_rows = [paired_row(ep, nominal[ep["seed"]]) for ep in fired]
            log_summary(group_rows)
            plot_group(task, perturbation, fired, nominal, run_dir / "figures")
            rows.extend(group_rows)

    with (run_dir / "eval" / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("wrote %s", run_dir)


def baseline_sigma(baseline: Path, task: str) -> np.ndarray:
    """Returns sigma from the successful M1 baseline episodes of one task."""
    actions = []
    for path in sorted((baseline / "baseline" / task).glob("ep_*.npz")):
        with np.load(path) as data:
            if bool(data["success"]):
                actions.append(data["action_exec"])
    return normalization_sigma(actions)


def load_episodes(run: Path, task: str, sigma: np.ndarray, settings: Settings) -> list[Episode]:
    """Loads one task's episodes of a run with their trigger features."""
    episodes = []
    for meta_path in sorted((run / "episodes" / task).glob("*/*/ep_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(meta_path.parent / "data.npz") as data:
            chunks, action_exec = data["chunks"], data["action_exec"]
        disagreement = group_disagreement(chunks, sigma, settings.robot, settings.trigger)
        episodes.append(
            {
                "id": meta["episode_id"],
                "seed": meta["seed"],
                "perturbation": meta["perturbation"],
                "perturb": meta["perturb_step"],
                "onset": meta["onset_step"],
                "failure": meta["failure_type"],
                "action_exec": action_exec,
                "d": disagreement.max(axis=1),
                "flips": gripper_flips(chunks, action_exec, settings.robot, settings.trigger),
            }
        )
    return episodes


def paired_row(ep: Episode, counterpart: Episode) -> dict[str, str | int | float | bool]:
    """Compares the window after the perturbation with the same steps of the counterpart."""
    start = int(ep["perturb"]) + 1
    window = ep["percentile"][start : start + WINDOW]
    reference_window = counterpart["percentile"][start : start + WINDOW]
    row = {
        "episode": ep["id"],
        "failure": ep["failure"],
        "perturb_step": int(ep["perturb"]),
        "onset_lag": int(ep["onset"]) - int(ep["perturb"]) if int(ep["onset"]) >= 0 else -1,
        "identical_prefix": bool(
            np.array_equal(ep["action_exec"][: start - 1], counterpart["action_exec"][: start - 1])
        ),
        "peak": round(float(window.max()), 4),
        "peak_nominal": round(float(reference_window.max()), 4),
        "flip_in_window": bool(ep["flips"][start : start + WINDOW].any()),
    }
    for level in LEVELS:
        crossings = np.flatnonzero(window >= level)
        row[f"first_{level}"] = int(crossings[0]) + 1 if len(crossings) else -1
        row[f"nominal_{level}"] = bool((reference_window >= level).any())
    return row


def log_summary(rows: list[dict[str, str | int | float | bool]]) -> None:
    """Logs crossing rates after the perturbation and on the counterparts' same steps."""
    for level in LEVELS:
        lags = [row[f"first_{level}"] for row in rows if row[f"first_{level}"] > 0]
        logging.info(
            "  level %.2f: perturbed %d/%d (median first crossing %s steps after the "
            "perturbation), unperturbed same steps %d/%d",
            level,
            len(lags),
            len(rows),
            f"{np.median(lags):.0f}" if lags else "-",
            sum(bool(row[f"nominal_{level}"]) for row in rows),
            len(rows),
        )
    onset_lags = [row["onset_lag"] for row in rows if row["onset_lag"] >= 0]
    logging.info(
        "  median onset lag %s; identical prefixes %d/%d",
        f"{np.median(onset_lags):.0f}" if onset_lags else "-",
        sum(bool(row["identical_prefix"]) for row in rows),
        len(rows),
    )


def plot_group(
    task: str,
    perturbation: str,
    group: list[Episode],
    nominal: dict[int, Episode],
    figures: Path,
) -> None:
    """Plots percentile d(t) aligned to the perturbation step, with a CSV twin."""
    offsets = np.arange(PLOT_RANGE[0], PLOT_RANGE[1] + 1)
    figure, axis = plt.subplots(figsize=(8, 4))
    curves, reference_curves = [], []
    for ep in group:
        steps = int(ep["perturb"]) + offsets
        valid = (steps >= 1) & (steps < len(ep["percentile"]))
        curve = np.full(len(offsets), np.nan)
        curve[valid] = ep["percentile"][steps[valid]]
        reference = np.full(len(offsets), np.nan)
        reference[valid] = nominal[ep["seed"]]["percentile"][steps[valid]]
        curves.append(curve)
        reference_curves.append(reference)
        axis.plot(offsets, curve, color="tab:blue", alpha=0.25, linewidth=1)
        if int(ep["onset"]) > int(ep["perturb"]):
            axis.axvline(int(ep["onset"]) - int(ep["perturb"]), color="tab:red", alpha=0.25)
    axis.plot(
        offsets,
        np.nanmedian(reference_curves, axis=0),
        color="grey",
        linewidth=2,
        label="median, same seeds unperturbed",
    )
    axis.plot(offsets, np.nanmedian(curves, axis=0), color="black", linewidth=2, label="median")
    axis.axvline(0, color="tab:orange", linewidth=2, label="perturbation")
    axis.axhline(LEVELS[0], color="grey", linestyle="--", linewidth=1)
    axis.set(
        xlabel="steps from perturbation (red: labeled onsets)",
        ylabel="d(t) percentile of unperturbed steps",
        title=f"{task} / {perturbation}: chunk disagreement (dev seeds, provisional)",
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize=8)
    stem = figures / f"disagreement_{task}_{perturbation}"
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(figure)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        ids = [str(ep["id"]) for ep in group]
        writer.writerow(["offset", *ids, *[f"{name} (unperturbed)" for name in ids]])
        columns = np.array([*curves, *reference_curves]).T
        writer.writerows([offset, *values] for offset, values in zip(offsets, columns, strict=True))


if __name__ == "__main__":
    main()
