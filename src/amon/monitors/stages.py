"""Monitor stages of a sweep: train the world model, then score the stored episodes.

Scores go to `scores/<monitor>/<episode_id>.npz` (`grid_steps`, `score`; spec section 2). Each
shard also writes `scores/<monitor>/meta_<task>_shard<i>.json` with the model and prompt hashes
and the latency measured on that shard's GPU. Finished score files are kept, so a stage can be
rerun after an interruption and only scores what is missing.
"""

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from amon.monitors.vlm import VlmJudge, load_prompt
from amon.monitors.wm import (
    device,
    encode_frames,
    load_encoder,
    load_predictor,
    measure_check_latency,
    save_predictor,
    score_episode,
    train_predictor,
)
from amon.rollout.store import EpisodeKey, episode_dir
from amon.settings import Settings

LOGGER = logging.getLogger(__name__)


def stored_keys(run_dir: Path, keys: list[EpisodeKey]) -> list[EpisodeKey]:
    """Returns the keys whose episode has been written completely."""
    return [key for key in keys if (episode_dir(run_dir, key) / "meta.json").exists()]


def score_path(run_dir: Path, monitor: str, key: EpisodeKey) -> Path:
    """Returns the score file of one episode."""
    return run_dir / "scores" / monitor / f"{key.episode_id}.npz"


def write_scores(path: Path, grid_steps: np.ndarray, score: np.ndarray) -> None:
    """Writes one episode's scores through a temporary file, so no half-written file remains."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"partial_{path.name}")
    with partial.open("wb") as handle:
        np.savez(handle, grid_steps=grid_steps.astype(np.int64), score=score.astype(np.float32))
    partial.replace(path)


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Returns the frames at (height, width) `size`, area-averaged; unchanged if already so."""
    if frames.shape[1:3] == tuple(size):
        return frames
    return np.stack(
        [cv2.resize(frame, size[::-1], interpolation=cv2.INTER_AREA) for frame in frames]
    )


def latency_summary(latencies_ms: np.ndarray) -> dict[str, float]:
    """Returns the mean and 95th percentile of measured latencies, ignoring failed requests."""
    valid = latencies_ms[~np.isnan(latencies_ms)]
    return {
        "mean": float(valid.mean()),
        "p95": float(np.percentile(valid, 95)),
        "count": int(len(valid)),
    }


def file_digest(path: Path) -> str:
    """Returns the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_wm(run_dir: Path, keys: list[EpisodeKey], settings: Settings) -> None:
    """Trains one world-model predictor per task that has unperturbed `train` episodes.

    Writes `models/wm_<task>.pt` and `eval/wm_train_<task>.json` (validation losses and the
    dimensions `score_wm` needs to load the model). A task whose model exists is skipped.
    """
    train_keys = [key for key in stored_keys(run_dir, keys) if key.split == "train"]
    tasks = list(dict.fromkeys(key.task for key in train_keys))
    if not tasks:
        LOGGER.info("train_wm: no stored train episodes, nothing to do")
        return
    encoder = load_encoder(settings.wm)
    (run_dir / "models").mkdir(exist_ok=True)
    for task in tasks:
        model_path = run_dir / "models" / f"wm_{task}.pt"
        if model_path.exists():
            LOGGER.info("train_wm: %s exists, kept", model_path.name)
            continue
        latents, actions = [], []
        for key in [key for key in train_keys if key.task == task]:
            with np.load(episode_dir(run_dir, key) / "data.npz") as data:
                frames = resize_frames(data["frames"], settings.wm.input_size)
                latents.append(encode_frames(encoder, frames, settings.wm.encode_batch))
                actions.append(data["action_exec"])
        LOGGER.info("train_wm: %s, %d episodes", task, len(latents))
        predictor, val_losses = train_predictor(latents, actions, settings.wm)
        save_predictor(model_path, predictor)
        summary = {
            "episodes": len(latents),
            "latent_dim": int(latents[0].shape[1]),
            "action_dim": int(actions[0].shape[1]),
            "val_losses": val_losses,
            "encoder": f"{settings.wm.encoder_repo}/{settings.wm.encoder}",
        }
        (run_dir / "eval" / f"wm_train_{task}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


def score_wm(run_dir: Path, keys: list[EpisodeKey], settings: Settings, shard: int) -> int:
    """Scores this shard's calib and test episodes with the world model of their task.

    Tasks without a trained model are skipped with a log line (insertion has no train split).

    Returns:
        The number of episodes that raised an error.
    """
    pending = [
        key
        for key in stored_keys(run_dir, keys)
        if key.split != "train" and not score_path(run_dir, "wm", key).exists()
    ]
    tasks = list(dict.fromkeys(key.task for key in pending))
    if not tasks:
        LOGGER.info("score_wm: nothing to score")
        return 0
    encoder = load_encoder(settings.wm)
    failed = 0
    for task in tasks:
        model_path = run_dir / "models" / f"wm_{task}.pt"
        if not model_path.exists():
            LOGGER.info("score_wm: no world model for %s, its episodes are not scored", task)
            continue
        dimensions = json.loads(
            (run_dir / "eval" / f"wm_train_{task}.json").read_text(encoding="utf-8")
        )
        predictor = load_predictor(
            model_path, dimensions["latent_dim"], dimensions["action_dim"], settings.wm
        )
        latencies = measure_check_latency(encoder, predictor, settings.wm)
        meta = {
            "monitor": "wm",
            "encoder": dimensions["encoder"],
            "model_sha256": file_digest(model_path),
            "grid_stride": settings.wm.grid_stride,
            "input_size": list(settings.wm.input_size),
            "gpu": torch.cuda.get_device_name() if device().type == "cuda" else "cpu",
            "latency_ms": latency_summary(latencies),
        }
        (run_dir / "scores" / "wm").mkdir(parents=True, exist_ok=True)
        (run_dir / "scores" / "wm" / f"meta_{task}_shard{shard}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        task_keys = [key for key in pending if key.task == task]
        for index, key in enumerate(task_keys, start=1):
            try:
                with np.load(episode_dir(run_dir, key) / "data.npz") as data:
                    frames = resize_frames(data["frames"], settings.wm.input_size)
                    latents = encode_frames(encoder, frames, settings.wm.encode_batch)
                    scores = score_episode(predictor, latents, data["action_exec"], settings.wm)
            except Exception:  # A broken episode must not stop the shard.
                LOGGER.exception("SCORE_FAILED wm %s", key.episode_id)
                failed += 1
                continue
            grid = np.arange(0, len(scores), settings.wm.grid_stride)
            write_scores(score_path(run_dir, "wm", key), grid, scores[grid])
            if index % 20 == 0 or index == len(task_keys):
                LOGGER.info("score_wm: %s %d/%d", task, index, len(task_keys))
    return failed


def score_vlm(
    run_dir: Path, keys: list[EpisodeKey], settings: Settings, endpoint: str | None, shard: int
) -> int:
    """Scores this shard's calib and test episodes with the VLM judge on its grid.

    Latency is measured first on `latency_sample` grid frames sent one at a time, as a deployed
    monitor would; the episodes are then scored with concurrent requests. An episode with any
    failed request is not written, so a rerun retries it (finished requests are cached).

    Args:
        run_dir: Run folder.
        keys: This shard's episodes.
        settings: Global settings.
        endpoint: Server URL ending in /v1; `monitors.vlm.endpoint` if None.
        shard: Shard index, for the meta and cache file names.

    Returns:
        The number of episodes that were not scored completely.
    """
    vlm_settings = replace(settings.vlm, endpoint=endpoint or settings.vlm.endpoint)
    pending = [
        key
        for key in stored_keys(run_dir, keys)
        if key.split != "train" and not score_path(run_dir, "vlm", key).exists()
    ]
    if not pending:
        LOGGER.info("score_vlm: nothing to score")
        return 0
    scores_dir = run_dir / "scores" / "vlm"
    scores_dir.mkdir(parents=True, exist_ok=True)
    server_version = vlm_server_version(vlm_settings.endpoint)
    failed = 0
    for task in dict.fromkeys(key.task for key in pending):
        prompt = load_prompt(task)
        judge = VlmJudge(
            vlm_settings, vlm_settings.model, prompt, scores_dir / f"cache_shard{shard}.jsonl"
        )
        task_keys = [key for key in pending if key.task == task]
        with np.load(episode_dir(run_dir, task_keys[0]) / "data.npz") as data:
            sample = data["frames"][:: vlm_settings.grid_stride][: vlm_settings.latency_sample]
        judge.request(sample[0])  # warm-up, not timed
        latencies = np.array([judge.request(frame)[1] for frame in sample])
        if np.isnan(latencies).all():
            LOGGER.error("score_vlm: the server at %s does not answer", vlm_settings.endpoint)
            return len(pending)
        meta = {
            "monitor": "vlm",
            "model": vlm_settings.model,
            "model_revision": model_snapshot(vlm_settings.model),
            "vllm_version": server_version,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "grid_stride": vlm_settings.grid_stride,
            "max_pixels": vlm_settings.max_pixels,
            "frame_size": list(sample.shape[1:3]),
            "latency_ms": latency_summary(latencies),
        }
        (scores_dir / f"meta_{task}_shard{shard}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info(
            "score_vlm: %s latency %.0f ms mean, %.0f ms p95",
            task,
            meta["latency_ms"]["mean"],
            meta["latency_ms"]["p95"],
        )
        for index, key in enumerate(task_keys, start=1):
            with np.load(episode_dir(run_dir, key) / "data.npz") as data:
                grid = np.arange(0, len(data["frames"]), vlm_settings.grid_stride)
                scores, _ = judge.score(data["frames"][grid])
            if np.isnan(scores).any():
                LOGGER.error(
                    "SCORE_FAILED vlm %s: %d of %d requests failed",
                    key.episode_id,
                    int(np.isnan(scores).sum()),
                    len(scores),
                )
                failed += 1
                continue
            write_scores(score_path(run_dir, "vlm", key), grid, scores)
            if index % 10 == 0 or index == len(task_keys):
                LOGGER.info("score_vlm: %s %d/%d", task, index, len(task_keys))
    return failed


def vlm_server_version(endpoint: str) -> str:
    """Returns the vLLM version the server reports, or "unknown" if it does not answer."""
    try:
        response = requests.get(endpoint.removesuffix("/v1") + "/version", timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        LOGGER.warning("could not read the vLLM version: %s", error)
        return "unknown"
    return str(response.json().get("version", "unknown"))


def model_snapshot(repo: str) -> str:
    """Returns the commit of the locally cached snapshot of a Hugging Face model."""
    try:
        path = snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        LOGGER.warning("%s is not in the local Hugging Face cache; revision unknown", repo)
        return "unknown"
    return Path(path).name
