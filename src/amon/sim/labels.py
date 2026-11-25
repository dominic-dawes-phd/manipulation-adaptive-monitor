"""Failure events and onset labels (spec section 4), as pure functions of logged arrays.

All state arrays are aligned with the frames: index t describes what frame t shows, before action
t is applied. `reward[t]` is the outcome of action t, so it first shows in frame t + 1.
"""

from dataclasses import dataclass

import numpy as np

from amon.settings import LabelSettings, RobotSettings
from amon.sim.state import ARMS

HANDOVER_STAGE = 3
INSERT_STAGE = 4
LIFT_STAGE = 2
# Object each arm grasps, by position in the task's object list.
GRASP_TARGETS = {"transfer_cube": (0, 0), "insertion": (1, 0)}


@dataclass(frozen=True)
class FailureEvent:
    """One labeled event."""

    kind: str
    onset: int


def phase_from_reward(reward: np.ndarray) -> np.ndarray:
    """Returns p_t, the highest reward stage reached before frame t, shape [T]."""
    highest = np.maximum.accumulate(reward)
    return np.concatenate([[0], highest[:-1]]).astype(np.int8)


def boundary_steps(phase: np.ndarray) -> np.ndarray:
    """Returns the steps at which the phase increases."""
    return np.flatnonzero(np.diff(phase) > 0) + 1


def find_drops(held: np.ndarray, debounce: int) -> list[FailureEvent]:
    """Finds drops: an object held by some arm is then held by none for `debounce` steps.

    Args:
        held: [T, K, 2] bool.
        debounce: Shortest gap that counts; contacts flicker off for a step or two in firm grasps.

    Returns:
        One event per drop, at the first step without a hold.
    """
    events = []
    steps = held.shape[0]
    for k in range(held.shape[1]):
        any_arm = held[:, k].any(axis=1)
        for t in np.flatnonzero(any_arm[:-1] & ~any_arm[1:]) + 1:
            if t + debounce <= steps and not any_arm[t : t + debounce].any():
                events.append(FailureEvent("drop", int(t)))
    return events


def close_commands(action_exec: np.ndarray, gripper: int, threshold: float) -> np.ndarray:
    """Returns the steps at which the commanded gripper crosses from open to closed."""
    is_open = action_exec[:, gripper] > threshold
    return np.flatnonzero(is_open[:-1] & ~is_open[1:]) + 1


def find_grasp_misses(
    task: str,
    contact: np.ndarray,
    action_exec: np.ndarray,
    robot: RobotSettings,
    window: int,
) -> list[FailureEvent]:
    """Finds close commands after which the arm never touches its object within `window` steps."""
    events = []
    steps = len(action_exec)
    for arm, target in enumerate(GRASP_TARGETS[task]):
        (gripper,) = robot.groups[f"{ARMS[arm]}_grip"]
        for t in close_commands(action_exec, gripper, robot.gripper_close_threshold):
            observed = contact[t : t + window + 1, target, arm]
            if t + window < steps and not observed.any():
                events.append(FailureEvent("grasp_miss", int(t)))
    return events


def find_stage_timeout(
    kind: str, phase: np.ndarray, start_stage: int, goal_stage: int, window: int, at_start: bool
) -> list[FailureEvent]:
    """Finds a stage that is not reached within `window` steps of an earlier stage.

    Args:
        kind: Failure type to report.
        phase: [T] highest stage reached.
        start_stage: Stage that starts the clock.
        goal_stage: Stage that must follow; any higher stage also counts.
        window: Allowed steps.
        at_start: Whether the onset is the start of the clock or the first late step.

    Returns:
        Zero or one event.
    """
    started = np.flatnonzero(phase >= start_stage)
    if len(started) == 0:
        return []
    start = int(started[0])
    deadline = start + window
    if deadline >= len(phase) or phase[deadline] >= goal_stage:
        return []
    return [FailureEvent(kind, start if at_start else deadline)]


def label_failure(
    task: str,
    success: bool,
    phase: np.ndarray,
    held: np.ndarray,
    contact: np.ndarray,
    action_exec: np.ndarray,
    labels: LabelSettings,
    robot: RobotSettings,
) -> FailureEvent:
    """Labels one episode with its earliest failure event.

    Returns:
        `none` for successes, `unlabeled` for failures without an event (both with onset -1).
    """
    if success:
        return FailureEvent("none", -1)
    events = find_drops(held, labels.drop_debounce_steps)
    events += find_grasp_misses(task, contact, action_exec, robot, labels.grasp_miss_window)
    if task == "transfer_cube":
        events += find_stage_timeout(
            "handover_miss", phase, LIFT_STAGE, HANDOVER_STAGE, labels.handover_window, False
        )
    else:
        events += find_stage_timeout(
            "insert_miss", phase, HANDOVER_STAGE, INSERT_STAGE, labels.insert_window, True
        )
    if not events:
        return FailureEvent("unlabeled", -1)
    return min(events, key=_onset_of)


def _onset_of(event: FailureEvent) -> int:
    """Sort key."""
    return event.onset
