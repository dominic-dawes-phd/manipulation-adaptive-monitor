"""M3 sanity gate for the VLM judge: AUROC on 50 labeled frames and ms per call.

`make` draws 25 failed frames from the perturbed dev review episodes (20 steps after a drop or
grasp_miss onset) and, as on-track frames, the same step of the same seed run unperturbed. Pairing
by seed and step keeps the task phase out of the comparison. It writes the frames, a shuffled
numbered contact sheet without labels, and labels.csv with the sim-proposed label and an empty
`label` column for the user's own judgement.

`score` sends the frames to a running server (`scripts/serve_vlm.sh`), once sequentially to
measure latency, and reports AUROC against `label` (the sim proposal where `label` is empty,
flagged as preliminary). Gate: AUROC >= monitors.vlm.sanity_auroc_gate (HANDOFF M3).

    python scripts/vlm_sanity.py make --review runs/review_<stamp> --nominal runs/<nominal>
    python scripts/vlm_sanity.py score --set runs/vlm_sanity_<stamp> [--model M] [--upscale 448]
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from amon.eval.stats import auroc
from amon.monitors.vlm import VlmJudge, load_prompt
from amon.rollout.manifest import write_manifest
from amon.settings import REPO_ROOT, load_settings

PAIRS = 25
FAILED_OFFSET = 20
SIM_EVENTS = ("drop", "grasp_miss")
SHEET_COLUMNS = 10
SHEET_CELL = 224
FIELDS = ["frame", "task", "episode", "step", "proposed", "label"]


def main() -> None:
    """Dispatches to `make` or `score`."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--review", type=Path, required=True)
    make.add_argument("--nominal", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--set", type=Path, required=True)
    score.add_argument("--model", default=None)
    score.add_argument("--upscale", type=int, default=0, help="resize frames to N x N first")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if arguments.command == "make":
        make_set(arguments.review, arguments.nominal)
    else:
        score_set(arguments.set, arguments.model, arguments.upscale)


def make_set(review: Path, nominal: Path) -> None:
    """Writes the paired frame set, the blind contact sheet and labels.csv."""
    settings = load_settings()
    run_dir = REPO_ROOT / "runs" / f"vlm_sanity_{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "frames").mkdir(parents=True)
    write_manifest(
        run_dir, {"name": "vlm_sanity", "review": str(review), "nominal": str(nominal)}, settings
    )
    groups: dict[str, list[tuple[Path, Path, int]]] = {}
    for meta_path in sorted((review / "episodes").glob("*/*/*/ep_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["failure_type"] not in SIM_EVENTS or meta["onset_step"] <= meta["perturb_step"]:
            continue
        counterpart = nominal / "episodes" / meta["task"] / "none" / meta["split"]
        counterpart = counterpart / meta_path.parent.name
        counterpart_meta = json.loads((counterpart / "meta.json").read_text(encoding="utf-8"))
        step = meta["onset_step"] + FAILED_OFFSET
        with np.load(counterpart / "data.npz") as data:
            counterpart_steps = len(data["reward"])
        still_on_track = counterpart_meta["success"] or counterpart_meta["onset_step"] > step
        if step < counterpart_steps and still_on_track:
            group = f"{meta['task']}/{meta['perturbation']}"
            groups.setdefault(group, []).append((meta_path.parent, counterpart, step))
    pairs = []
    while len(pairs) < PAIRS and any(groups.values()):
        for group in groups.values():
            if group and len(pairs) < PAIRS:
                pairs.append(group.pop(0))
    logging.info("%d pairs", len(pairs))

    rows, images = [], []
    for failed_dir, on_track_dir, step in pairs:
        for folder, proposed in ((failed_dir, "failed"), (on_track_dir, "on_track")):
            with np.load(folder / "data.npz") as data:
                images.append(data["frames"][step])
            rows.append(
                {
                    "task": folder.parts[-4],
                    "episode": "/".join(folder.parts[-4:]),
                    "step": step,
                    "proposed": proposed,
                    "label": "",
                }
            )
    order = np.random.default_rng(0).permutation(len(rows))
    sheet_cells = []
    for number, index in enumerate(order, start=1):
        rows[index]["frame"] = f"{number:02d}"
        imageio.imwrite(run_dir / "frames" / f"{number:02d}.png", images[index])
        cell = cv2.resize(images[index], (SHEET_CELL, SHEET_CELL), interpolation=cv2.INTER_CUBIC)
        cv2.putText(cell, f"{number:02d}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        sheet_cells.append(cell)
    blank = np.zeros_like(sheet_cells[0])
    sheet_cells += [blank] * (-len(sheet_cells) % SHEET_COLUMNS)
    sheet_rows = [
        np.hstack(sheet_cells[start : start + SHEET_COLUMNS])
        for start in range(0, len(sheet_cells), SHEET_COLUMNS)
    ]
    imageio.imwrite(run_dir / "sheet.png", np.vstack(sheet_rows))
    with (run_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=_frame_number))
    logging.info("wrote %s; fill the `label` column (failed / on_track) from sheet.png", run_dir)


def score_set(set_dir: Path, model: str | None, upscale: int) -> None:
    """Scores the set sequentially, then logs and writes AUROC and latency."""
    settings = load_settings(overrides={"monitors.vlm.concurrency": 1})
    served_model = model or settings.vlm.model
    with (set_dir / "labels.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hand_labeled = all(row["label"] in ("failed", "on_track") for row in rows)
    if not hand_labeled:
        logging.warning("PRELIMINARY: `label` is incomplete, using the sim-proposed labels")
    failed = np.array(
        [(row["label"] if hand_labeled else row["proposed"]) == "failed" for row in rows]
    )
    frames = []
    for row in rows:
        frame = imageio.imread(set_dir / "frames" / f"{row['frame']}.png")
        if upscale:
            frame = cv2.resize(frame, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
    scores = np.full(len(rows), np.nan)
    latencies = np.full(len(rows), np.nan)
    for task in sorted({row["task"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["task"] == task]
        judge = VlmJudge(settings.vlm, served_model, load_prompt(task), set_dir / "vlm_cache.jsonl")
        scores[indices], latencies[indices] = judge.score(np.stack([frames[i] for i in indices]))
    valid = ~np.isnan(scores)
    area = auroc(scores[valid], failed[valid])
    measured = latencies[~np.isnan(latencies)]
    size = upscale or frames[0].shape[0]
    logging.info(
        "%s @%dpx: AUROC %.3f on %d frames (%d without an answer)%s; latency %s ms per call; "
        "gate %.2f -> %s",
        served_model,
        size,
        area,
        int(valid.sum()),
        int((~valid).sum()),
        "" if hand_labeled else " PRELIMINARY",
        f"median {np.median(measured):.0f}, p90 {np.percentile(measured, 90):.0f}"
        if len(measured)
        else "- (all cached)",
        settings.vlm.sanity_auroc_gate,
        "PASS" if area >= settings.vlm.sanity_auroc_gate else "FAIL",
    )
    out = set_dir / f"scores_{served_model.replace('/', '_')}_{size}px.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "task", "failed", "score", "latency_ms"])
        for row, is_failed, value, latency in zip(rows, failed, scores, latencies, strict=True):
            writer.writerow([row["frame"], row["task"], bool(is_failed), value, latency])


def _frame_number(row: dict[str, str | int]) -> str:
    """Sort key."""
    return str(row["frame"])


if __name__ == "__main__":
    main()
