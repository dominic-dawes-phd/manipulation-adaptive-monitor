"""VLM judge scoring, caching and AUROC, without a server."""

import math
from pathlib import Path

import numpy as np
import pytest
import requests

from amon.eval.stats import auroc
from amon.monitors import vlm
from amon.settings import load_settings

SETTINGS = load_settings()


class FakeResponse:
    """A chat completion whose first token is Yes with log-probability log(0.8)."""

    def raise_for_status(self) -> None:
        """Never fails."""

    def json(self) -> dict:
        """Returns the OpenAI-compatible logprobs layout vLLM sends."""
        alternatives = [
            {"token": "Yes", "logprob": math.log(0.8)},
            {"token": "No", "logprob": math.log(0.15)},
            {"token": " no", "logprob": math.log(0.05)},
        ]
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": alternatives}]}}]}


def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
    """Stands in for requests.post and counts calls."""
    fake_post.calls += 1
    return FakeResponse()


def failing_post(url: str, json: dict, timeout: float) -> FakeResponse:
    """Stands in for an unreachable server."""
    raise requests.ConnectionError("refused")


def test_no_probability_pools_variants_and_handles_absence() -> None:
    """Case and whitespace variants are pooled; neither answer gives NaN."""
    alternatives = [
        {"token": "Yes", "logprob": math.log(0.6)},
        {"token": " yes", "logprob": math.log(0.1)},
        {"token": "No", "logprob": math.log(0.3)},
    ]
    assert vlm.no_probability(alternatives) == pytest.approx(0.3)
    assert math.isnan(vlm.no_probability([{"token": "Maybe", "logprob": 0.0}]))


def test_judge_scores_once_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated frames are scored once; a new judge reads the cache from disk."""
    fake_post.calls = 0
    monkeypatch.setattr(vlm.requests, "post", fake_post)
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    frames[1] = 255
    cache = tmp_path / "cache.jsonl"
    judge = vlm.VlmJudge(SETTINGS.vlm, "model", "prompt", cache)
    scores, latencies = judge.score(frames)
    np.testing.assert_allclose(scores, 0.2)
    assert fake_post.calls == 2  # frames 0 and 2 are identical
    assert not np.isnan(latencies[:2]).any()
    reopened = vlm.VlmJudge(SETTINGS.vlm, "model", "prompt", cache)
    scores_again, latencies_again = reopened.score(frames)
    np.testing.assert_allclose(scores_again, 0.2)
    assert fake_post.calls == 2 and np.isnan(latencies_again).all()
    other_prompt = vlm.VlmJudge(SETTINGS.vlm, "model", "another prompt", cache)
    other_prompt.score(frames[:1])
    assert fake_post.calls == 3


def test_failed_requests_give_nan_and_are_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection error is logged, scored NaN and retried next time."""
    monkeypatch.setattr(vlm.requests, "post", failing_post)
    judge = vlm.VlmJudge(SETTINGS.vlm, "model", "prompt", tmp_path / "cache.jsonl")
    scores, _ = judge.score(np.zeros((1, 8, 8, 3), dtype=np.uint8))
    assert np.isnan(scores).all() and not judge.cache


def test_auroc_counts_ties_as_half() -> None:
    """Perfect, inverted and tied rankings."""
    labels = np.array([False, False, True, True])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
