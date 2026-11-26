"""Trigger features on synthetic chunk streams."""

from dataclasses import replace

import numpy as np

from amon.settings import load_settings
from amon.trigger.features import gripper_flips, group_disagreement, normalization_sigma

SETTINGS = load_settings()
STEPS, HORIZON, DIM = 120, 100, 14
RIGHT_ARM_COLUMN = list(SETTINGS.robot.groups).index("right_arm")
RIGHT_GRIP = SETTINGS.robot.groups["right_grip"][0]


def _chunks_following(plan: np.ndarray) -> np.ndarray:
    """Chunks that predict the given plan exactly: A_t[j] = plan[t + j], held at the end."""
    steps = len(plan)
    rows = np.minimum(np.arange(steps)[:, None] + np.arange(HORIZON)[None, :], steps - 1)
    return plan[rows]


def _smooth_plan() -> np.ndarray:
    """A slowly varying 14-D trajectory with open grippers."""
    plan = np.sin(np.linspace(0.0, 3.0, STEPS))[:, None] * np.ones(DIM, dtype=np.float32)
    plan[:, [6, 13]] = 1.0
    return plan.astype(np.float32)


def test_consistent_chunks_have_zero_disagreement() -> None:
    """Chunks that agree on their overlap give zero everywhere."""
    chunks = _chunks_following(_smooth_plan())
    values = group_disagreement(chunks, np.ones(DIM), SETTINGS.robot, SETTINGS.trigger)
    assert values.shape == (STEPS, len(SETTINGS.robot.groups))
    np.testing.assert_allclose(values, 0.0, atol=1e-6)


def test_plan_change_shows_once_in_its_group() -> None:
    """A persistent right-arm plan change from step 50 spikes only that group, only at 50."""
    chunks = _chunks_following(_smooth_plan())
    chunks[50:, :, 7:13] += 0.3
    values = group_disagreement(chunks, np.ones(DIM), SETTINGS.robot, SETTINGS.trigger)
    assert np.flatnonzero(values.max(axis=1) > 1e-6).tolist() == [50]
    assert values[50].argmax() == RIGHT_ARM_COLUMN
    np.testing.assert_allclose(values[50, RIGHT_ARM_COLUMN], 0.3 * np.sqrt(6), rtol=1e-5)


def test_sigma_scales_and_window_limits_disagreement() -> None:
    """Dividing by sigma rescales; a change beyond the window M is not seen."""
    plan = _smooth_plan()
    chunks = _chunks_following(plan)
    chunks[50:, :, 7:13] += 0.3
    sigma = np.ones(DIM)
    sigma[7:13] = 3.0
    values = group_disagreement(chunks, sigma, SETTINGS.robot, SETTINGS.trigger)
    np.testing.assert_allclose(values[50, RIGHT_ARM_COLUMN], 0.1 * np.sqrt(6), rtol=1e-5)
    late = _chunks_following(plan)
    late[50:, SETTINGS.trigger.disagreement_window + 1 :, 7:13] += 0.3
    late_values = group_disagreement(late, np.ones(DIM), SETTINGS.robot, SETTINGS.trigger)
    np.testing.assert_allclose(late_values, 0.0, atol=1e-6)


def test_normalization_sigma_pools_episodes() -> None:
    """Sigma is the standard deviation over all steps of all episodes."""
    first = np.zeros((10, DIM))
    second = np.full((10, DIM), 2.0)
    np.testing.assert_allclose(normalization_sigma([first, second]), 1.0)


def test_gripper_flip_fires_once_per_predicted_event() -> None:
    """A close at 60 and a reopen at 70 each fire on the first step that predicts them."""
    plan = _smooth_plan()
    plan[60:70, RIGHT_GRIP] = 0.0
    plan[80:, RIGHT_GRIP] = 0.0
    chunks = _chunks_following(plan)
    lookahead = SETTINGS.trigger.gripper_lookahead
    fired = gripper_flips(chunks, plan, SETTINGS.robot, SETTINGS.trigger)
    # The close at 60 first enters the lookahead at 60 - lookahead + 1; the reopen at 70 is
    # predicted from 61, once the close has been executed. The close at 80 falls within the
    # refractory period of the first close.
    assert np.flatnonzero(fired).tolist() == [60 - lookahead + 1, 61]
    short = replace(SETTINGS.trigger, gripper_refractory=10)
    fired_short = gripper_flips(chunks, plan, SETTINGS.robot, short)
    assert np.flatnonzero(fired_short).tolist() == [51, 61, 71]
