"""Cheap trigger features from logged ACT chunks (spec section 8).

All functions are causal: the value at step t uses only chunks queried at or before t and actions
executed before t.
"""

import numpy as np

from amon.settings import RobotSettings, TriggerSettings

GRIPPER_GROUPS = ("left_grip", "right_grip")


def normalization_sigma(action_sequences: list[np.ndarray]) -> np.ndarray:
    """Returns the per-dimension standard deviation of executed actions.

    Args:
        action_sequences: `action_exec` arrays [T, D] of the episodes to pool (calib successes).

    Returns:
        Array [D].
    """
    return np.concatenate(action_sequences).std(axis=0)


def group_disagreement(
    chunks: np.ndarray, sigma: np.ndarray, robot: RobotSettings, trigger: TriggerSettings
) -> np.ndarray:
    """Returns e_g(t), the disagreement of consecutive shadow chunks per action group.

    Compares A_t[j] with A_{t-s}[j+s] over the first M' = min(M, H - s) overlapping steps,
    normalized by sigma, as the mean of per-step L2 norms within each group.

    Args:
        chunks: Shadow chunks [T, H, D].
        sigma: Normalization [D] from `normalization_sigma`.
        robot: Supplies the action groups; columns follow `robot.groups` order.
        trigger: Supplies the window M and stride s.

    Returns:
        Array [T, G]; the first s steps have no earlier chunk and are zero.
    """
    stride = trigger.disagreement_stride
    window = min(trigger.disagreement_window, chunks.shape[1] - stride)
    scaled = chunks.astype(np.float32) / sigma.astype(np.float32)
    difference = scaled[stride:, :window] - scaled[:-stride, stride : stride + window]
    values = np.zeros((len(chunks), len(robot.groups)), dtype=np.float32)
    for column, indices in enumerate(robot.groups.values()):
        norms = np.linalg.norm(difference[:, :, list(indices)], axis=2)
        values[stride:, column] = norms.mean(axis=1)
    return values


def gripper_flips(
    chunks: np.ndarray, action_exec: np.ndarray, robot: RobotSettings, trigger: TriggerSettings
) -> np.ndarray:
    """Returns g(t): an imminent gripper open/close predicted by the chunk queried at t.

    For each arm and direction, a flip is predicted at t when the first `gripper_lookahead` steps
    of A_t cross the open/close threshold relative to the last executed command (step t - 1). The
    feature fires on the first step of each predicted flip, at most once per
    `gripper_refractory` steps for the same arm and direction.

    Args:
        chunks: Shadow chunks [T, H, D].
        action_exec: Executed actions [T, D].
        robot: Supplies the gripper indices and the threshold.
        trigger: Supplies the lookahead and the refractory period.

    Returns:
        Boolean array [T]; step 0 never fires.
    """
    threshold = robot.gripper_close_threshold
    fired = np.zeros(len(chunks), dtype=bool)
    for group in GRIPPER_GROUPS:
        (index,) = robot.groups[group]
        was_open = action_exec[:-1, index] > threshold
        predicted_open = chunks[1:, : trigger.gripper_lookahead, index] > threshold
        closing = was_open & ~predicted_open.all(axis=1)
        opening = ~was_open & predicted_open.any(axis=1)
        for predicted in (closing, opening):
            starts = np.flatnonzero(predicted & ~np.concatenate(([False], predicted[:-1])))
            last_fire = -trigger.gripper_refractory
            for start in starts:
                if start - last_fire >= trigger.gripper_refractory:
                    fired[start + 1] = True
                    last_fire = start
    return fired
