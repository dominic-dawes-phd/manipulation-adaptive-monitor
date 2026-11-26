"""M4 acceptance on dev seeds: does the world-model score rise after a failure onset?

Encodes the frames of a run of unperturbed dev episodes, trains the predictor on the seeds from
TRAIN_FIRST_SEED on, and scores the perturbed review episodes and their unperturbed counterparts
(same seeds, identical up to the perturbation; found in the same unperturbed run). Writes the
onset-aligned mean scores (PNG and CSV), the predictor and a summary. Dev only, never reported.

    python scripts/wm_dev_check.py --nominal runs/wm_dev_<stamp> --review runs/review_<stamp>
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from amon.monitors.wm import (
    device,
    encode_frames,
    load_encoder,
    save_predictor,
    score_episode,
    train_predictor,
)
from amon.rollout.manifest import write_manifest
from amon.settings import REPO_ROOT, Settings, load_settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)

TASK = "transfer_cube"
TRAIN_FIRST_SEED = 2010
WINDOW = 20
PLOT_RANGE = (-30, 60)
SIM_EVENTS = ("drop", "grasp_miss")
LATENCY_REPEATS = 50


def main() -> None:
    """Trains, scores and summarizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    settings = load_settings()
    run_dir = REPO_ROOT / "runs" / f"wm_check_{time.strftime('%Y%m%d-%H%M%S')}"
    for folder in ("figures", "eval", "models"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    spec = {"name": "wm_check", "nominal": str(arguments.nominal), "review": str(arguments.review)}
    write_manifest(run_dir, spec, settings)
    encoder = load_encoder(settings.wm)

    nominal: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for data_path in sorted(
        (arguments.nominal / "episodes" / TASK / "none").glob("*/ep_*/data.npz")
    ):
        with np.load(data_path) as data:
            latents = encode_frames(encoder, data["frames"], settings.wm.encode_batch)
            nominal[int(data_path.parent.name[3:])] = (latents, data["action_exec"])
    train_seeds = [seed for seed in nominal if seed >= TRAIN_FIRST_SEED]
    logging.info("%d training episodes", len(train_seeds))
    predictor, val_losses = train_predictor(
        [nominal[seed][0] for seed in train_seeds],
        [nominal[seed][1] for seed in train_seeds],
        settings.wm,
    )
    save_predictor(run_dir / "models" / "predictor.pt", predictor)

    groups: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    for meta_path in sorted((arguments.review / "episodes" / TASK).glob("*/*/ep_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["failure_type"] not in SIM_EVENTS or meta["onset_step"] <= meta["perturb_step"]:
            continue
        counterpart_steps = len(nominal[meta["seed"]][0])
        if counterpart_steps < meta["onset_step"] + WINDOW:
            logging.info(
                "skipped %s: counterpart ends at %d", meta["episode_id"], counterpart_steps
            )
            continue
        with np.load(meta_path.parent / "data.npz") as data:
            latents = encode_frames(encoder, data["frames"], settings.wm.encode_batch)
            scores = score_episode(predictor, latents, data["action_exec"], settings.wm)
        counterpart_latents, counterpart_actions = nominal[meta["seed"]]
        counterpart = score_episode(
            predictor, counterpart_latents, counterpart_actions, settings.wm
        )
        groups.setdefault(meta["perturbation"], []).append(
            (scores, counterpart, meta["onset_step"])
        )

    rows = []
    offsets = np.arange(PLOT_RANGE[0], PLOT_RANGE[1] + 1)
    figure, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.5), squeeze=False)
    for axis, (perturbation, episodes) in zip(axes[0], sorted(groups.items()), strict=True):
        curves = np.full((2, len(episodes), len(offsets)), np.nan)
        higher = 0
        for index, (scores, counterpart, onset) in enumerate(episodes):
            steps = onset + offsets
            for row, series in enumerate((scores, counterpart)):
                valid = (steps >= 0) & (steps < len(series))
                curves[row, index, valid] = series[steps[valid]]
            after = slice(onset, onset + WINDOW)
            higher += int(np.nanmax(scores[after]) > np.nanmax(counterpart[after]))
        means = np.nanmean(curves, axis=1)
        axis.plot(offsets, means[1], color="grey", linewidth=2, label="same seeds unperturbed")
        axis.plot(offsets, means[0], color="black", linewidth=2, label="perturbed")
        axis.axvline(0, color="tab:red", label="onset")
        axis.set(title=f"{TASK} / {perturbation} (n={len(episodes)})", xlabel="steps from onset")
        axis.set_ylabel("mean 1 - cos(ẑ, z)")
        axis.legend(fontsize=7)
        window = slice(-PLOT_RANGE[0], -PLOT_RANGE[0] + WINDOW)
        rows.append(
            {
                "perturbation": perturbation,
                "episodes": len(episodes),
                "mean_after_onset": float(np.nanmean(curves[0][:, window])),
                "mean_counterpart": float(np.nanmean(curves[1][:, window])),
                "peak_higher_than_counterpart": higher,
            }
        )
        logging.info(
            "%s: mean score in the %d steps after onset %.4f vs %.4f unperturbed; peak higher "
            "in %d/%d",
            perturbation,
            WINDOW,
            rows[-1]["mean_after_onset"],
            rows[-1]["mean_counterpart"],
            higher,
            len(episodes),
        )
        with (run_dir / "figures" / f"wm_score_{perturbation}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["offset", "perturbed_mean", "unperturbed_mean"])
            writer.writerows(zip(offsets, means[0], means[1], strict=True))
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "wm_score_after_onset.png", dpi=120)
    plt.close(figure)

    latency_ms = check_latency(encoder, predictor, nominal[train_seeds[0]], settings)
    summary = {"val_losses": val_losses, "latency_ms_per_check": latency_ms, "groups": rows}
    (run_dir / "eval" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("latency %.1f ms per check on %s; wrote %s", latency_ms, device(), run_dir)


def check_latency(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    episode: tuple[np.ndarray, np.ndarray],
    settings: Settings,
) -> float:
    """Returns the median ms to encode one frame and score it, as in a live check."""
    latents, actions = episode
    height, width = settings.render.store_size
    frame = np.zeros((1, height, width, 3), dtype=np.uint8)
    durations = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        encode_frames(encoder, frame, 1)
        score_episode(predictor, latents[:13], actions[:13], settings.wm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(durations))


if __name__ == "__main__":
    main()
