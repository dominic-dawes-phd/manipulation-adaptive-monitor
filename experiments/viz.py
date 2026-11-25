"""Review reels: labeled clips around each perturbation, one MP4 per perturbation.

Quick M2 version. `python -m experiments.viz --run runs/review_<stamp>` writes
figures/review_<task>_<perturbation>.mp4 and a matching contact sheet. Needs episodes recorded
with video.
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

LEAD_STEPS = 30
TAIL_STEPS = 70
SHEET_OFFSETS = (-6, -1, 0, 2, 5, 10, 20, 40)
PLAYBACK_FPS = 20  # 0.4x real time
VIEW_SIZE = (960, 360)
WHITE, RED, ORANGE, GREY = (255, 255, 255), (255, 70, 70), (255, 165, 0), (150, 150, 150)
Meta = dict[str, str | int | bool | list[int] | dict[str, float]]


def main() -> None:
    """Writes one reel and one contact sheet per task and perturbation found in the run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    figures = arguments.run / "figures"
    figures.mkdir(exist_ok=True)
    groups: dict[tuple[str, str], list[Path]] = {}
    for meta_path in sorted((arguments.run / "episodes").glob("*/*/*/ep_*/meta.json")):
        task, perturbation = meta_path.parts[-5], meta_path.parts[-4]
        groups.setdefault((task, perturbation), []).append(meta_path.parent)
    for (task, perturbation), episodes in groups.items():
        stem = figures / f"review_{task}_{perturbation}"
        write_reel(episodes, stem.with_suffix(".mp4"))
        write_sheet(episodes, stem.with_suffix(".png"))


def write_reel(episodes: list[Path], out: Path) -> None:
    """Concatenates a clip around the event of every episode."""
    with imageio.get_writer(
        out, fps=PLAYBACK_FPS, macro_block_size=1, ffmpeg_log_level="error"
    ) as writer:
        for number, folder in enumerate(episodes, start=1):
            meta, held, video = _load(folder)
            center = _center_step(meta, len(video))
            last = max(center, int(meta["onset_step"])) + TAIL_STEPS
            for step in range(max(0, center - LEAD_STEPS), min(len(video), last)):
                writer.append_data(_annotate(video[step], meta, held, step, number))


def write_sheet(episodes: list[Path], out: Path) -> None:
    """Draws one row per episode, with columns at fixed offsets from the event."""
    rows = []
    for number, folder in enumerate(episodes, start=1):
        meta, held, video = _load(folder)
        center = _center_step(meta, len(video))
        cells = []
        for offset in SHEET_OFFSETS:
            step = int(np.clip(center + offset, 0, len(video) - 1))
            angle_view = video[step][:, video[step].shape[1] // 2 :]
            cell = cv2.resize(angle_view, (240, 180), interpolation=cv2.INTER_AREA)
            hold = "held" if held[step].any() else "NOT HELD"
            color = RED if 0 <= int(meta["onset_step"]) <= step else WHITE
            _text(cell, f"#{number} t={step} ({offset:+d}) {hold}", 14, color, 0.42)
            cells.append(cell)
        rows.append(np.hstack(cells))
    imageio.imwrite(out, np.vstack(rows))


def _load(folder: Path) -> tuple[Meta, np.ndarray, list[np.ndarray]]:
    """Reads the metadata, the hold flags and the video of one episode."""
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "data.npz") as data:
        held = data["held"]
    with imageio.get_reader(folder / "video.mp4") as reader:
        video = [frame for frame in reader]
    return meta, held, video


def _center_step(meta: Meta, steps: int) -> int:
    """Returns the event to show: the perturbation, else the onset, else mid-episode."""
    if int(meta["perturb_step"]) >= 0:
        return int(meta["perturb_step"])
    return int(meta["onset_step"]) if int(meta["onset_step"]) >= 0 else steps // 2


def _annotate(
    frame: np.ndarray, meta: Meta, held: np.ndarray, step: int, number: int
) -> np.ndarray:
    """Scales a frame and overlays the labels."""
    image = cv2.resize(frame, VIEW_SIZE, interpolation=cv2.INTER_AREA)
    fired, onset = int(meta["perturb_step"]), int(meta["onset_step"])
    outcome = "success" if meta["success"] else f"fail: {meta['failure_type']}"
    _text(image, f"#{number} {meta['perturbation']} seed {meta['seed']} -> {outcome}", 20, WHITE)
    _text(image, f"t={step}  perturb={fired}  onset={onset}", 42, WHITE)
    if fired >= 0 and step >= fired:
        _text(image, "PERTURBED", 336, ORANGE)
    if onset >= 0 and step >= onset:
        _text(image, f"{'':14}ONSET +{step - onset}", 336, RED)
    _text(image, "held" if held[step].any() else "not held", 64, GREY)
    if step in (fired, onset):
        color = RED if step == onset else ORANGE
        cv2.rectangle(image, (0, 0), (VIEW_SIZE[0] - 1, VIEW_SIZE[1] - 1), color, 8)
    return image


def _text(
    image: np.ndarray, text: str, y: int, color: tuple[int, int, int], scale: float = 0.55
) -> None:
    """Draws one line of text."""
    cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
