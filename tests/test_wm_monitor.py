"""World-model predictor, windows and scores on synthetic latents (no encoder download)."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from amon.monitors.wm import (
    load_predictor,
    prediction_windows,
    save_predictor,
    score_episode,
    train_predictor,
)
from amon.settings import load_settings

SETTINGS = replace(load_settings().wm, epochs=15, hidden_dim=64, batch_size=64)
STEPS, LATENT_DIM, ACTION_DIM = 120, 16, 3


def _synthetic_episodes(count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Latents that drift in the direction of the executed actions."""
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(ACTION_DIM, LATENT_DIM)).astype(np.float32)
    latents, actions = [], []
    for _ in range(count):
        action = np.cumsum(rng.normal(scale=0.1, size=(STEPS, ACTION_DIM)), axis=0)
        state = np.ones(LATENT_DIM, dtype=np.float32)
        trajectory = []
        for step in range(STEPS):
            trajectory.append(state.copy())
            state = state + 0.05 * action[step].astype(np.float32) @ mixing
        latents.append(np.stack(trajectory))
        actions.append(action.astype(np.float32))
    return latents, actions


def test_prediction_windows_use_only_the_past() -> None:
    """Step t is predicted from z[t-k-L+1 .. t-k] and a[t-k .. t-1]."""
    latents = np.arange(30, dtype=np.float32)[:, None]
    actions = 100 + np.arange(30, dtype=np.float32)[:, None]
    steps, window_latents, window_actions = prediction_windows(latents, actions, 4, 8)
    assert steps[0] == 11 and steps[-1] == 29
    assert window_latents[0, :, 0].tolist() == [0, 1, 2, 3]
    assert window_actions[0, :, 0].tolist() == list(range(103, 111))
    assert window_latents[-1, -1, 0] == 29 - 8


def test_training_learns_and_scores_are_causal(tmp_path: Path) -> None:
    """Validation loss falls; later frames do not change earlier scores; save/load round trip."""
    latents, actions = _synthetic_episodes(20)
    predictor, val_losses = train_predictor(latents, actions, SETTINGS)
    assert val_losses[-1] < 0.5 * val_losses[0]
    scores = score_episode(predictor, latents[0], actions[0], SETTINGS)
    first = SETTINGS.history + SETTINGS.horizon - 1
    assert np.isnan(scores[:first]).all() and not np.isnan(scores[first:]).any()
    altered = latents[0].copy()
    altered[80:] *= -1.0
    altered_scores = score_episode(predictor, altered, actions[0], SETTINGS)
    np.testing.assert_array_equal(altered_scores[:80], scores[:80])
    assert altered_scores[80] > 10 * np.nanmedian(scores)
    path = tmp_path / "predictor.pt"
    save_predictor(path, predictor)
    reloaded = load_predictor(path, LATENT_DIM, ACTION_DIM, SETTINGS)
    np.testing.assert_allclose(
        score_episode(reloaded, latents[0], actions[0], SETTINGS), scores, rtol=1e-5, atol=1e-7
    )
