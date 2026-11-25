"""Configuration consistency and interval arithmetic."""

import numpy as np
import pytest

from amon.eval.stats import exact_mcnemar, wilson_interval
from amon.settings import load_settings


def test_action_groups_follow_gym_aloha_order() -> None:
    """The configured groups match the joint order declared by gym-aloha."""
    from gym_aloha.constants import ACTIONS

    groups = load_settings().robot.groups
    for arm in ("left", "right"):
        assert all(ACTIONS[i].startswith(f"{arm}_arm_") for i in groups[f"{arm}_arm"])
        (gripper_index,) = groups[f"{arm}_grip"]
        assert ACTIONS[gripper_index] == f"{arm}_arm_gripper"


def test_split_seed_ranges() -> None:
    """Evaluation splits keep their documented seed ranges."""
    splits = load_settings().splits
    assert splits["train"] == (0, 500)
    assert splits["calib"] == (10000, 10300)
    assert splits["test"] == (20000, 20300)
    assert splits["baseline"] == (1000, 1200)


def test_wilson_interval_known_value() -> None:
    """Matches the textbook value for 8 of 10."""
    result = wilson_interval(8, 10, 0.95)
    assert result.estimate == pytest.approx(0.8)
    assert result.low == pytest.approx(0.4902, abs=1e-4)
    assert result.high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_edges() -> None:
    """The interval stays inside [0, 1] at the extremes."""
    assert wilson_interval(0, 20, 0.95).low == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(20, 20, 0.95).high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0, 0.95)


def test_exact_mcnemar() -> None:
    """Counts discordant pairs and tests them with a two-sided binomial."""
    first = np.array([True] * 5 + [False] * 5)
    second = np.array([False] * 5 + [False] * 5)
    result = exact_mcnemar(first, second)
    assert (result.only_first, result.only_second) == (5, 0)
    assert result.p_value == pytest.approx(2 * 0.5**5)
    assert exact_mcnemar(first, first).p_value == 1.0
