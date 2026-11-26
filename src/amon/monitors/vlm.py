"""VLM judge: P("No" | "Is the task on track?") from first-token logprobs (spec section 6.1).

Talks to a vLLM server through its OpenAI-compatible chat API (`scripts/serve_vlm.sh`, run in
the separate `.venv-vlm` environment, D-08). Scores are cached by (model, prompt, frame).
"""

import base64
import hashlib
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

from amon.settings import REPO_ROOT, VlmSettings

PROMPT_DIR = REPO_ROOT / "configs" / "prompts"
ANSWERS = ("yes", "no")


def load_prompt(task: str) -> str:
    """Returns the judge prompt of a task."""
    return (PROMPT_DIR / f"vlm_judge_{task}.txt").read_text(encoding="utf-8").strip()


def no_probability(top_logprobs: list[dict[str, str | float]]) -> float:
    """Returns P(No) / (P(Yes) + P(No)) over the first-token alternatives.

    Tokens that differ only in case or surrounding whitespace ("No", " no") are pooled.

    Args:
        top_logprobs: The `top_logprobs` entries of the first generated token.

    Returns:
        The score in [0, 1], or NaN if neither answer is among the alternatives.
    """
    mass = dict.fromkeys(ANSWERS, 0.0)
    for alternative in top_logprobs:
        word = str(alternative["token"]).strip().lower()
        if word in mass:
            mass[word] += math.exp(float(alternative["logprob"]))
    total = mass["yes"] + mass["no"]
    return mass["no"] / total if total > 0 else math.nan


def frame_digest(frame: np.ndarray) -> str:
    """Returns a content hash of an RGB frame, including its shape."""
    return hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()


class VlmJudge:
    """Scores RGB frames with one model and one prompt."""

    def __init__(self, settings: VlmSettings, model: str, prompt: str, cache_path: Path) -> None:
        """Opens the score cache.

        Args:
            settings: Endpoint, decoding and concurrency parameters.
            model: Served model id, for example `settings.model` or `settings.fallback_model`.
            prompt: Judge prompt, from `load_prompt`.
            cache_path: JSON-lines file of earlier scores; created if missing.
        """
        self.settings = settings
        self.model = model
        self.prompt = prompt
        self.cache_path = cache_path
        self.prefix = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:16]
        self.cache: dict[str, float] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["score"]

    def score(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scores frames, reusing cached scores.

        Args:
            frames: [N, H, W, 3] uint8.

        Returns:
            Scores [N] (NaN where a request failed or gave neither answer) and the latency of
            each request in ms [N] (NaN for cache hits and failures).
        """
        keys = [f"{self.prefix}:{frame_digest(frame)}" for frame in frames]
        scores = np.array([self.cache.get(key, math.nan) for key in keys])
        latencies = np.full(len(frames), math.nan)
        # Each distinct uncached frame is sent once; duplicates share its result.
        positions: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            positions.setdefault(key, []).append(index)
        missing = [key for key in positions if key not in self.cache]
        with ThreadPoolExecutor(max_workers=self.settings.concurrency) as pool:
            results = list(pool.map(self.request, [frames[positions[key][0]] for key in missing]))
        new_entries = []
        for key, (score, latency_ms) in zip(missing, results, strict=True):
            scores[positions[key]] = score
            latencies[positions[key][0]] = latency_ms
            if not math.isnan(score):
                self.cache[key] = score
                new_entries.append(json.dumps({"key": key, "score": score}))
        if new_entries:
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(new_entries) + "\n")
        return scores, latencies

    def request(self, frame: np.ndarray) -> tuple[float, float]:
        """Sends one frame; returns its score and the request latency in ms (NaN on failure)."""
        _, png = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        image_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": 1,
            "temperature": self.settings.temperature,
            "logprobs": True,
            "top_logprobs": self.settings.top_logprobs,
        }
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.settings.endpoint}/chat/completions",
                json=payload,
                timeout=self.settings.request_timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            logging.warning("VLM request failed: %s", error)
            return math.nan, math.nan
        latency_ms = 1000.0 * (time.perf_counter() - start)
        first_token = response.json()["choices"][0]["logprobs"]["content"][0]
        return no_probability(first_token["top_logprobs"]), latency_ms
