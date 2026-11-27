"""Perturbation mechanics and sweep expansion."""

from collections.abc import Iterator

import numpy as np
import pytest

from amon.rollout.sweep import expand_cells
from amon.settings import load_settings
from amon.sim.env import AlohaSim
from amon.sim.perturb import PERTURBATIONS, make_perturbation
from amon.sim.state import RIGHT, SimState

SETTINGS = load_settings()
SEED = 2000


@pytest.fixture(scope="module")
def sim() -> Iterator[AlohaSim]:
    """Provides a transfer_cube environment."""
    environment = AlohaSim(SETTINGS.tasks["transfer_cube"], SETTINGS.render)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def insertion_sim() -> Iterator[AlohaSim]:
    """Provides an insertion environment."""
    environment = AlohaSim(SETTINGS.tasks["insertion"], SETTINGS.render)
    yield environment
    environment.close()


def _state(held: bool, height: float = 0.1, objects: int = 1) -> SimState:
    """A state in which the right arm does or does not hold the first object."""
    flags = np.zeros((objects, 2), dtype=bool)
    flags[0, RIGHT] = held
    positions = np.zeros((objects, 3))
    positions[:, 2] = height
    return SimState(positions, flags, flags, np.ones((2, 3)))


def test_none_has_no_perturbation() -> None:
    """Nominal episodes get nothing."""
    assert make_perturbation("none", SETTINGS.tasks["transfer_cube"], SETTINGS, 0) is None


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_sampling_is_reproducible(name: str) -> None:
    """The same seed samples the same parameters."""
    task = SETTINGS.tasks["insertion"]
    first = make_perturbation(name, task, SETTINGS, SEED)
    second = make_perturbation(name, task, SETTINGS, SEED)
    assert first.parameters == second.parameters


def test_knock_fires_once_after_its_delay_and_clears(sim: AlohaSim) -> None:
    """The force lasts exactly the configured number of steps and is horizontal."""
    task = SETTINGS.tasks["transfer_cube"]
    sim.reset(SEED)
    knock = make_perturbation("knock", task, SETTINGS, SEED)
    delay = int(knock.parameters["delay"])
    force = sim.physics.named.data.xfrc_applied
    for step in range(100):
        knock.before_step(sim.physics, step, _state(held=step >= 10), phase=2)
        applied = np.linalg.norm(force["box"][:3])
        active = 10 + delay <= step < 10 + delay + SETTINGS.perturbations["knock"].steps
        assert applied == pytest.approx(SETTINGS.perturbations["knock"].force_n if active else 0.0)
        assert force["box"][2] == 0.0
    assert knock.step_fired == 10 + delay


def test_knock_stops_when_the_fingers_lose_the_object(sim: AlohaSim) -> None:
    """The push ends after two substeps without finger contact (D-18)."""
    task = SETTINGS.tasks["transfer_cube"]
    sim.reset(SEED)
    knock = make_perturbation("knock", task, SETTINGS, SEED)
    step = 0
    while knock.step_fired < 0:
        knock.before_step(sim.physics, step, _state(held=True), phase=2)
        step += 1
    force = sim.physics.named.data.xfrc_applied["box"]
    assert knock.substep_hook_needed()
    knock.substep(sim.physics)  # at reset the cube touches no finger
    assert np.linalg.norm(force[:3]) > 0
    knock.substep(sim.physics)
    assert not knock.substep_hook_needed()
    assert np.linalg.norm(force[:3]) == 0.0
    knock.before_step(sim.physics, step, _state(held=True), phase=2)
    assert np.linalg.norm(force[:3]) == 0.0


def test_knock_needs_a_hold_in_stage_2(sim: AlohaSim) -> None:
    """No hold, or another stage, means no knock."""
    task = SETTINGS.tasks["transfer_cube"]
    sim.reset(SEED)
    knock = make_perturbation("knock", task, SETTINGS, SEED)
    for step in range(60):
        knock.before_step(sim.physics, step, _state(held=False), phase=2)
        knock.before_step(sim.physics, step, _state(held=True), phase=1)
        knock.before_step(sim.physics, step, _state(held=True), phase=3)
    assert knock.step_fired == -1


def test_slip_changes_contact_friction_and_restores_it(sim: AlohaSim) -> None:
    """The finger gets priority, otherwise MuJoCo would use the cube's larger friction."""
    task = SETTINGS.tasks["transfer_cube"]
    sim.reset(SEED)
    model = sim.physics.named.model
    geom = SETTINGS.robot.finger_geoms["right"][0]
    friction = model.geom_friction[geom].copy()
    slip = make_perturbation("slip", task, SETTINGS, SEED)
    for step in range(100):
        slip.before_step(sim.physics, step, _state(True, height=0.1 + 0.001 * step), phase=2)
        elapsed = step - slip.step_fired
        if slip.step_fired >= 0 and elapsed < SETTINGS.perturbations["slip"].steps:
            scale = SETTINGS.perturbations["slip"].friction_scale
            assert np.allclose(model.geom_friction[geom], friction * scale)
            assert model.geom_priority[geom] == 1
        else:
            assert np.array_equal(model.geom_friction[geom], friction)
            assert model.geom_priority[geom] == 0
    assert slip.step_fired >= 0


def test_slip_restores_when_the_episode_ends_early(sim: AlohaSim) -> None:
    """The friction change must not leak into the next episode."""
    task = SETTINGS.tasks["transfer_cube"]
    sim.reset(SEED)
    model = sim.physics.named.model
    geom = SETTINGS.robot.finger_geoms["right"][0]
    slip = make_perturbation("slip", task, SETTINGS, SEED)
    step = 0
    while slip.step_fired < 0:
        slip.before_step(sim.physics, step, _state(True, height=0.1 + 0.001 * step), phase=2)
        step += 1
    assert model.geom_friction[geom][0] < 1.0
    slip.restore(sim.physics)
    assert model.geom_friction[geom][0] == 1.0
    assert model.geom_priority[geom] == 0


def test_nudge_moves_the_object_by_the_offset(sim: AlohaSim) -> None:
    """The cube moves horizontally by the configured distance and stays on the table."""
    task = SETTINGS.tasks["transfer_cube"]
    observation = sim.reset(SEED)
    for _ in range(40):
        sim.step_without_rendering(observation.agent_pos)
    before = np.array(sim.physics.named.data.qpos["red_box_joint"])
    nudge = make_perturbation("nudge", task, SETTINGS, SEED)
    flags = np.zeros((1, 2), dtype=bool)
    near = SimState(before[None, :3], flags, flags, np.array([before[:3], before[:3]]))
    nudge.before_step(sim.physics, 40, near, phase=0)
    after = np.array(sim.physics.named.data.qpos["red_box_joint"])
    assert nudge.step_fired == 40
    shift = np.linalg.norm(after[:2] - before[:2])
    assert shift == pytest.approx(SETTINGS.perturbations["nudge"].offset_m)
    assert after[2] == pytest.approx(before[2])


def test_peg_rotation_matches_the_sampled_angle(insertion_sim: AlohaSim) -> None:
    """The peg turns by the sampled angle and keeps its distance to the pinch point."""
    task = SETTINGS.tasks["insertion"]
    insertion_sim.reset(SEED)
    data = insertion_sim.physics.named.data
    fingers = [np.array(data.geom_xpos[g]) for g in SETTINGS.robot.finger_geoms["right"]]
    pinch = (fingers[0] + fingers[1]) / 2
    before = np.array(data.qpos["red_peg_joint"])
    peg = make_perturbation("peg_misalign", task, SETTINGS, SEED)
    step = 0
    while peg.step_fired < 0:
        peg.before_step(insertion_sim.physics, step, _state(True, objects=2), phase=2)
        step += 1
    after = np.array(data.qpos["red_peg_joint"])
    cosine = min(1.0, abs(float(before[3:] @ after[3:])))
    turned = np.rad2deg(2 * np.arccos(cosine))
    assert turned == pytest.approx(abs(peg.parameters["angle_deg"]), abs=1e-3)
    radius_before = np.linalg.norm(before[:3] - pinch)
    assert np.linalg.norm(after[:3] - pinch) == pytest.approx(radius_before)


def test_expand_cells_uses_disjoint_seed_ranges() -> None:
    """Each cell draws from its own range."""
    spec = {
        "tasks": ["transfer_cube", "insertion"],
        "cells": [
            {"perturbation": "none", "split": "test", "episodes": 2},
            {"perturbation": "slip", "split": "test", "episodes": 1, "tasks": ["transfer_cube"]},
        ],
    }
    ids = [key.episode_id for key in expand_cells(spec, SETTINGS)]
    assert ids == [
        "transfer_cube/none/test/ep_020000",
        "transfer_cube/none/test/ep_020001",
        "transfer_cube/slip/test/ep_031000",
        "insertion/none/test/ep_020000",
        "insertion/none/test/ep_020001",
    ]
    spec["only"] = [ids[2]]
    assert [key.episode_id for key in expand_cells(spec, SETTINGS)] == [ids[2]]


def test_expand_cells_rejects_bad_cells() -> None:
    """Too many seeds, or a perturbed calibration cell, is an error."""
    big = {"perturbation": "none", "split": "test", "episodes": 100000}
    leak = {"perturbation": "knock", "split": "calib", "episodes": 1}
    for cell in (big, leak):
        with pytest.raises(ValueError):
            expand_cells({"tasks": ["transfer_cube"], "cells": [cell]}, SETTINGS)
