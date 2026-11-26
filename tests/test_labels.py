"""Failure labels on synthetic streams."""

import numpy as np

from amon.settings import load_settings
from amon.sim.labels import (
    FailureEvent,
    boundary_steps,
    find_drops,
    find_grasp_misses,
    find_stage_timeout,
    label_failure,
    phase_from_reward,
)

SETTINGS = load_settings()
STEPS = 300


def _held(*intervals: tuple[int, int], steps: int = STEPS) -> np.ndarray:
    """One object, held by the right arm during the given half-open intervals."""
    held = np.zeros((steps, 1, 2), dtype=bool)
    for start, end in intervals:
        held[start:end, 0, 1] = True
    return held


def _open_gripper_actions(steps: int = STEPS) -> np.ndarray:
    """Actions with both grippers open throughout."""
    actions = np.zeros((steps, 14), dtype=np.float32)
    actions[:, [6, 13]] = 1.0
    return actions


def test_phase_is_frame_aligned_and_monotonic() -> None:
    """A reward earned by action t first shows in frame t + 1, and regressions are ignored."""
    reward = np.array([0, 1, 0, 2, 1, 4], dtype=np.int8)
    phase = phase_from_reward(reward)
    assert phase.tolist() == [0, 0, 1, 1, 2, 2]
    assert boundary_steps(phase).tolist() == [2, 4]


def test_drop_onset_is_first_step_without_hold() -> None:
    """The onset is the first step without a hold."""
    assert find_drops(_held((50, 120)), 5) == [FailureEvent("drop", 120)]


def test_contact_flicker_is_not_a_drop() -> None:
    """Gaps shorter than the debounce are ignored; a longer one counts."""
    assert find_drops(_held((50, 80), (82, STEPS)), 5) == []
    assert find_drops(_held((50, 80), (85, STEPS)), 5) == [FailureEvent("drop", 80)]


def test_handover_is_not_a_drop() -> None:
    """Passing the object from one arm to the other is not a drop."""
    held = _held((50, 100))
    held[98:, 0, 0] = True
    assert find_drops(held, 5) == []


def test_grasp_miss_uses_the_close_command_step() -> None:
    """A close with no contact in the window is a miss at the close step."""
    actions = _open_gripper_actions()
    actions[60:, 13] = 0.0
    contact = np.zeros((STEPS, 1, 2), dtype=bool)
    robot = SETTINGS.robot
    missed = find_grasp_misses("transfer_cube", contact, actions, robot, 10)
    assert missed == [FailureEvent("grasp_miss", 60)]
    contact[70, 0, 1] = True
    assert find_grasp_misses("transfer_cube", contact, actions, robot, 10) == []
    contact[70, 0, 1] = False
    contact[71, 0, 1] = True
    assert find_grasp_misses("transfer_cube", contact, actions, robot, 10) == missed


def test_insertion_arms_have_their_own_targets() -> None:
    """The left arm must touch the socket (object 1), not the peg (object 0)."""
    actions = _open_gripper_actions()
    actions[60:, 6] = 0.0
    contact = np.zeros((STEPS, 2, 2), dtype=bool)
    contact[60:, 0, 0] = True
    misses = find_grasp_misses("insertion", contact, actions, SETTINGS.robot, 10)
    assert misses == [FailureEvent("grasp_miss", 60)]
    contact[60:, 1, 0] = True
    assert find_grasp_misses("insertion", contact, actions, SETTINGS.robot, 10) == []


def test_handover_timeout_accepts_a_skipped_stage() -> None:
    """Going straight from stage 2 to 4 is a good handover (D-16)."""
    phase = np.zeros(STEPS, dtype=np.int8)
    phase[40:] = 2
    late = find_stage_timeout("handover_miss", phase, 2, 3, 200, False)
    assert late == [FailureEvent("handover_miss", 240)]
    phase[240:] = 4
    assert find_stage_timeout("handover_miss", phase, 2, 3, 200, False) == []


def test_timeout_needs_the_full_window() -> None:
    """An episode that ends before the deadline is not a timeout."""
    phase = np.zeros(STEPS, dtype=np.int8)
    phase[150:] = 2
    assert find_stage_timeout("handover_miss", phase, 2, 3, 200, False) == []


def test_insert_miss_onset_is_the_stage_3_step() -> None:
    """The weak label is placed at the start of the clock."""
    phase = np.zeros(STEPS, dtype=np.int8)
    phase[100:] = 3
    event = find_stage_timeout("insert_miss", phase, 3, 4, 150, True)
    assert event == [FailureEvent("insert_miss", 100)]


def test_earliest_event_labels_the_episode() -> None:
    """The earliest event wins; successes and eventless failures have no onset."""
    phase = np.zeros(STEPS, dtype=np.int8)
    phase[50:] = 2
    held = _held((50, 120))
    contact = held.copy()
    arguments = (
        phase,
        held,
        contact,
        _open_gripper_actions(),
        SETTINGS.labels,
        SETTINGS.robot,
        "none",
        -1,
    )
    assert label_failure("transfer_cube", False, *arguments) == FailureEvent("drop", 120)
    assert label_failure("transfer_cube", True, *arguments) == FailureEvent("none", -1)
    nothing = np.zeros_like(held)
    empty = (np.zeros_like(phase), nothing, nothing, *arguments[3:])
    assert label_failure("transfer_cube", False, *empty) == FailureEvent("unlabeled", -1)


def test_perturbed_failures_are_attributed_by_timing() -> None:
    """Q8: an event before the perturbation; Q7: no event soon after it; else the event wins."""
    phase = np.zeros(STEPS, dtype=np.int8)
    phase[50:] = 2
    held = _held((50, 120))
    base = (phase, held, held.copy(), _open_gripper_actions(), SETTINGS.labels, SETTINGS.robot)
    window = SETTINGS.labels.perturbation_event_window
    expected = {
        120: FailureEvent("failed_before_perturbation", 120),
        110: FailureEvent("drop", 120),
        119 - window: FailureEvent("knock", 120 - window),
        -1: FailureEvent("drop", 120),
    }
    for perturb_step, event in expected.items():
        assert label_failure("transfer_cube", False, *base, "knock", perturb_step) == event
