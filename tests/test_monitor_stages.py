"""Monitor stages on a tiny synthetic run folder (no encoder download, no VLM server)."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import requests
from torch import nn

from amon.monitors import stages, vlm
from amon.rollout.store import EpisodeKey, episode_dir
from amon.settings import Settings, load_settings

BASE = load_settings()
SETTINGS = replace(
    BASE,
    wm=replace(BASE.wm, input_size=(4, 4), epochs=2, hidden_dim=16, batch_size=32, grid_stride=2),
    vlm=replace(BASE.vlm, grid_stride=5, latency_sample=3, concurrency=2),
)
STEPS, SIDE = 40, 8
KEYS = [EpisodeKey("transfer_cube", "none", "train", seed) for seed in range(6)] + [
    EpisodeKey("transfer_cube", "none", "test", 20000),
    EpisodeKey("transfer_cube", "knock", "test", 30000),
    EpisodeKey("insertion", "none", "test", 20000),
]


def fake_encode(encoder: nn.Module, frames: np.ndarray, batch_size: int) -> np.ndarray:
    """Uses the first 12 pixel values of each frame as its latent."""
    return frames.reshape(len(frames), -1)[:, :12].astype(np.float32) / 255.0 + 1.0


def fake_load_encoder(settings: Settings) -> nn.Module:
    """Stands in for the torch.hub download."""
    return nn.Identity()


def fake_latency(encoder: nn.Module, predictor: nn.Module, settings: Settings) -> np.ndarray:
    """Two fixed check durations."""
    return np.array([1.0, 3.0])


class YesResponse:
    """A chat completion that answers Yes with probability 0.75."""

    def raise_for_status(self) -> None:
        """Never fails."""

    def json(self) -> dict:
        """Returns the first-token logprobs layout vLLM sends."""
        alternatives = [
            {"token": "Yes", "logprob": math.log(0.75)},
            {"token": "No", "logprob": math.log(0.25)},
        ]
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": alternatives}]}}]}


def yes_post(url: str, json: dict, timeout: float) -> YesResponse:
    """Stands in for requests.post."""
    return YesResponse()


def refused(url: str, **kwargs: float) -> YesResponse:
    """Stands in for an unreachable server."""
    raise requests.ConnectionError("refused")


def snapshot_name(repo: str) -> str:
    """Stands in for the Hugging Face cache lookup."""
    return "abc123"


def _write_run(run_dir: Path) -> None:
    """Writes the episodes of KEYS with random frames and actions."""
    rng = np.random.default_rng(0)
    for key in KEYS:
        folder = episode_dir(run_dir, key)
        folder.mkdir(parents=True)
        np.savez_compressed(
            folder / "data.npz",
            frames=rng.integers(0, 255, size=(STEPS, SIDE, SIDE, 3), dtype=np.uint8),
            action_exec=rng.normal(size=(STEPS, 14)).astype(np.float32),
        )
        (folder / "meta.json").write_text(json.dumps({"episode_id": key.episode_id}))
    (run_dir / "eval").mkdir()


def test_resize_frames_keeps_matching_frames_and_area_averages() -> None:
    """Frames already at the size come back unchanged; 8x8 to 4x4 averages 2x2 blocks."""
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[:, :2, :2] = 200
    assert stages.resize_frames(frames, (8, 8)) is frames
    resized = stages.resize_frames(frames, (4, 4))
    assert resized.shape == (2, 4, 4, 3) and resized[0, 0, 0, 0] == 200 and resized[0, 1, 1, 0] == 0


def test_wm_stages_train_score_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One model per task with train episodes; calib/test scored on the grid; reruns skip."""
    monkeypatch.setattr(stages, "encode_frames", fake_encode)
    monkeypatch.setattr(stages, "load_encoder", fake_load_encoder)
    monkeypatch.setattr(stages, "measure_check_latency", fake_latency)
    _write_run(tmp_path)
    stages.train_wm(tmp_path, KEYS, SETTINGS)
    assert (tmp_path / "models" / "wm_transfer_cube.pt").exists()
    assert not (tmp_path / "models" / "wm_insertion.pt").exists()
    summary = json.loads((tmp_path / "eval" / "wm_train_transfer_cube.json").read_text())
    assert summary["latent_dim"] == 12 and len(summary["val_losses"]) == 2

    assert stages.score_wm(tmp_path, KEYS, SETTINGS, shard=0) == 0
    with np.load(stages.score_path(tmp_path, "wm", KEYS[6])) as scores:
        np.testing.assert_array_equal(scores["grid_steps"], np.arange(0, STEPS, 2))
        first = SETTINGS.wm.history + SETTINGS.wm.horizon - 1
        assert np.isnan(scores["score"][scores["grid_steps"] < first]).all()
        assert not np.isnan(scores["score"][scores["grid_steps"] >= first]).any()
    assert not stages.score_path(tmp_path, "wm", KEYS[0]).exists()  # train is not scored
    assert not stages.score_path(tmp_path, "wm", KEYS[8]).exists()  # no insertion model
    meta = json.loads((tmp_path / "scores" / "wm" / "meta_transfer_cube_shard0.json").read_text())
    assert meta["latency_ms"]["mean"] == 2.0

    written = stages.score_path(tmp_path, "wm", KEYS[6]).stat().st_mtime_ns
    stages.train_wm(tmp_path, KEYS, SETTINGS)
    assert stages.score_wm(tmp_path, KEYS, SETTINGS, shard=0) == 0
    assert stages.score_path(tmp_path, "wm", KEYS[6]).stat().st_mtime_ns == written


def test_vlm_stage_scores_grid_and_records_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every calib/test episode of every task is scored at the VLM grid stride."""
    monkeypatch.setattr(vlm.requests, "post", yes_post)
    monkeypatch.setattr(stages.requests, "get", refused)
    monkeypatch.setattr(stages, "model_snapshot", snapshot_name)
    _write_run(tmp_path)
    assert stages.score_vlm(tmp_path, KEYS, SETTINGS, "http://host:1/v1", shard=3) == 0
    for key in KEYS[6:]:
        with np.load(stages.score_path(tmp_path, "vlm", key)) as scores:
            np.testing.assert_array_equal(scores["grid_steps"], np.arange(0, STEPS, 5))
            np.testing.assert_allclose(scores["score"], 0.25)
    meta = json.loads((tmp_path / "scores" / "vlm" / "meta_insertion_shard3.json").read_text())
    assert meta["vllm_version"] == "unknown" and meta["model_revision"] == "abc123"
    assert meta["latency_ms"]["count"] == 3 and meta["frame_size"] == [SIDE, SIDE]


def test_vlm_stage_writes_nothing_when_the_server_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All pending episodes count as failed and no score file is written."""
    monkeypatch.setattr(vlm.requests, "post", refused)
    monkeypatch.setattr(stages.requests, "get", refused)
    monkeypatch.setattr(stages, "model_snapshot", snapshot_name)
    _write_run(tmp_path)
    assert stages.score_vlm(tmp_path, KEYS, SETTINGS, None, shard=0) == 3
    assert not list((tmp_path / "scores" / "vlm").rglob("*.npz"))
