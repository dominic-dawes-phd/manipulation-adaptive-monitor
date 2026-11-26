"""Interval estimates and paired tests."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import binomtest, norm


@dataclass(frozen=True)
class Proportion:
    """A proportion with its Wilson score interval."""

    successes: int
    trials: int
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedOutcome:
    """Discordant counts of two binary outcomes on the same episodes."""

    only_first: int
    only_second: int
    p_value: float


def wilson_interval(successes: int, trials: int, confidence: float) -> Proportion:
    """Computes the Wilson score interval of a proportion.

    Args:
        successes: Number of successes.
        trials: Number of trials, at least 1.
        confidence: Two-sided confidence level, for example 0.95.

    Returns:
        The estimate and its interval.
    """
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"invalid counts: {successes}/{trials}")
    z = float(norm.ppf(0.5 + confidence / 2))
    estimate = successes / trials
    scale = 1 + z**2 / trials
    center = (estimate + z**2 / (2 * trials)) / scale
    margin = z * sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2)) / scale
    return Proportion(
        successes, trials, estimate, max(0.0, center - margin), min(1.0, center + margin)
    )


def exact_mcnemar(first: np.ndarray, second: np.ndarray) -> PairedOutcome:
    """Runs the exact McNemar test on two paired boolean arrays."""
    if first.shape != second.shape:
        raise ValueError(f"shapes differ: {first.shape} vs {second.shape}")
    only_first = int(np.sum(first & ~second))
    only_second = int(np.sum(~first & second))
    discordant = only_first + only_second
    p_value = float(binomtest(only_first, discordant).pvalue) if discordant else 1.0
    return PairedOutcome(only_first, only_second, p_value)
