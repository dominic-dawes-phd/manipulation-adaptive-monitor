"""Perturbations that fire at most once per episode (spec section 5).

A perturbation sees the privileged state and edits only the physics, never the policy. The window
opens at the first step its condition holds. It fires U{0, delay_max} steps later, at the first
such step where the condition still holds.
"""

from abc import ABC, abstractmethod

import numpy as np
from dm_control import mujoco

from amon.settings import PerturbationSettings, Settings, TaskSettings
from amon.sim.state import RIGHT, SimState

PERTURBATIONS = ("knock", "slip", "nudge", "peg_misalign")
HELD_STAGE = 2
TARGET = 0  # first configured object: the cube, or the peg
NUDGE_ATTEMPTS = 32
RELEASE_SUBSTEPS = 2  # substeps without contact that count as released


class Perturbation(ABC):
    """Shared timing; subclasses define the window and the mechanism."""

    def __init__(
        self, settings: PerturbationSettings, task: TaskSettings, rng: np.random.Generator
    ) -> None:
        """Samples the delay."""
        self._task = task
        self._rng = rng
        self._delay = int(rng.integers(0, settings.delay_max + 1))
        self._earliest: int | None = None
        self.step_fired = -1
        self.parameters: dict[str, float] = {"delay": float(self._delay)}

    def before_step(self, physics: mujoco.Physics, step: int, state: SimState, phase: int) -> None:
        """Fires if due and advances an effect that is already under way."""
        if self.step_fired < 0 and self._window_open(state, phase):
            if self._earliest is None:
                self._earliest = step + self._delay
            if step >= self._earliest and self._fire(physics, state):
                self.step_fired = step
        if self.step_fired >= 0:
            self._advance(physics, step - self.step_fired)

    def substep(self, physics: mujoco.Physics) -> None:  # noqa: B027  (optional hook)
        """Called before each physics substep while `substep_hook_needed` is true."""

    def substep_hook_needed(self) -> bool:
        """Whether the effect must be updated faster than the control rate."""
        return False

    def restore(self, physics: mujoco.Physics) -> None:  # noqa: B027  (optional hook)
        """Undoes changes that would outlive the episode."""

    @abstractmethod
    def _window_open(self, state: SimState, phase: int) -> bool:
        """Whether the perturbation may fire in this state."""

    @abstractmethod
    def _fire(self, physics: mujoco.Physics, state: SimState) -> bool:
        """Starts the effect; returns False if it could not be applied yet."""

    def _advance(self, physics: mujoco.Physics, elapsed: int) -> None:  # noqa: B027
        """Continues an effect that lasts several steps."""

    def _target_held(self, state: SimState, phase: int) -> bool:
        """Whether the right arm holds the target and no handover or insertion has begun."""
        return phase == HELD_STAGE and bool(state.held[TARGET, RIGHT])

    def _target(self) -> str:
        """Name of the first configured object."""
        return next(iter(self._task.objects))


class Knock(Perturbation):
    """Pushes the held object sideways until it leaves the fingers.

    The push stops at the physics substep in which the fingers lose the object. A push of fixed
    duration would keep accelerating a 50 g object after the grip breaks and throw it off the
    table (D-18).
    """

    def __init__(
        self, settings: PerturbationSettings, task: TaskSettings, rng: np.random.Generator
    ) -> None:
        """Samples the push direction."""
        super().__init__(settings, task, rng)
        self._steps = settings.steps
        self._force_n = settings.force_n
        self._direction = settings.direction
        self._fingers = settings.finger_geoms
        target = task.objects[self._target()]
        self._body = target.body
        self._object_geoms = target.geoms
        self._angle = float(rng.uniform(0.0, 2 * np.pi))
        self._sign = 1.0 if rng.integers(0, 2) else -1.0
        self._force = np.zeros(3)
        self._pushing = False
        self._free_substeps = 0
        self.parameters.update(force_n=settings.force_n)

    def _window_open(self, state: SimState, phase: int) -> bool:
        return self._target_held(state, phase)

    def _fire(self, physics: mujoco.Physics, state: SimState) -> bool:
        unit = np.array([np.cos(self._angle), np.sin(self._angle), 0.0])
        if self._direction == "across_fingers":
            # The fingers block pushes along the axis they close on, so push across it.
            positions = [np.array(physics.named.data.geom_xpos[geom]) for geom in self._fingers]
            closing = positions[1] - positions[0]
            across = np.array([-closing[1], closing[0], 0.0])
            unit = self._sign * across / np.linalg.norm(across)
        self._force = self._force_n * unit
        self._pushing = True
        self.parameters.update(angle_rad=float(np.arctan2(unit[1], unit[0])))
        return True

    def _advance(self, physics: mujoco.Physics, elapsed: int) -> None:
        if elapsed >= self._steps:
            self._pushing = False
        physics.named.data.xfrc_applied[self._body][:3] = self._force if self._pushing else 0.0

    def substep(self, physics: mujoco.Physics) -> None:
        """Ends the push once the fingers have lost the object."""
        if not self._pushing:
            return
        self._free_substeps = 0 if self._touched(physics) else self._free_substeps + 1
        if self._free_substeps >= RELEASE_SUBSTEPS:
            self._pushing = False
            physics.named.data.xfrc_applied[self._body][:3] = 0.0

    def substep_hook_needed(self) -> bool:
        """Only while pushing."""
        return self._pushing

    def restore(self, physics: mujoco.Physics) -> None:
        """Removes the force."""
        physics.named.data.xfrc_applied[self._body][:3] = 0.0

    def _touched(self, physics: mujoco.Physics) -> bool:
        """Whether any right finger touches the object now."""
        model = physics.model
        objects = {model.name2id(geom, "geom") for geom in self._object_geoms}
        fingers = {model.name2id(geom, "geom") for geom in self._fingers}
        for contact in physics.data.contact[: physics.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & objects and pair & fingers:
                return True
        return False


class Slip(Perturbation):
    """Makes the right fingers slippery for a while.

    MuJoCo uses the larger friction of two touching geoms unless one has a higher priority, so
    lowering only the finger friction would have no effect. The fingers also get priority.
    """

    def __init__(
        self, settings: PerturbationSettings, task: TaskSettings, rng: np.random.Generator
    ) -> None:
        """Stores the mechanism's parameters."""
        super().__init__(settings, task, rng)
        self._steps = settings.steps
        self._scale = settings.friction_scale
        self._geoms = settings.finger_geoms
        self._last_height: float | None = None
        self._saved: list[tuple[np.ndarray, int]] = []
        self.parameters.update(friction_scale=settings.friction_scale)

    def before_step(self, physics: mujoco.Physics, step: int, state: SimState, phase: int) -> None:
        """Tracks the height change, which the window condition needs."""
        super().before_step(physics, step, state, phase)
        self._last_height = float(state.obj_pos[TARGET, 2])

    def _window_open(self, state: SimState, phase: int) -> bool:
        if self._last_height is None:
            return False
        return self._target_held(state, phase) and state.obj_pos[TARGET, 2] > self._last_height

    def _fire(self, physics: mujoco.Physics, state: SimState) -> bool:
        model = physics.named.model
        for geom in self._geoms:
            self._saved.append((model.geom_friction[geom].copy(), int(model.geom_priority[geom])))
            model.geom_friction[geom] = model.geom_friction[geom] * self._scale
            model.geom_priority[geom] = 1
        return True

    def _advance(self, physics: mujoco.Physics, elapsed: int) -> None:
        if elapsed >= self._steps:
            self.restore(physics)

    def restore(self, physics: mujoco.Physics) -> None:
        """Puts the original friction and priority back."""
        model = physics.named.model
        for geom, (friction, priority) in zip(self._geoms, self._saved, strict=False):
            model.geom_friction[geom] = friction
            model.geom_priority[geom] = priority
        self._saved = []


class Nudge(Perturbation):
    """Moves the object on the table just before the right gripper reaches it."""

    def __init__(
        self, settings: PerturbationSettings, task: TaskSettings, rng: np.random.Generator
    ) -> None:
        """Stores the mechanism's parameters."""
        super().__init__(settings, task, rng)
        self._radius = settings.trigger_radius_m
        self._offset = settings.offset_m
        self._joint = task.objects[self._target()].joint
        self.parameters.update(offset_m=settings.offset_m)

    def _window_open(self, state: SimState, phase: int) -> bool:
        gap = np.linalg.norm(state.gripper_pos[RIGHT] - state.obj_pos[TARGET])
        return phase == 0 and not state.contact[TARGET].any() and bool(gap < self._radius)

    def _fire(self, physics: mujoco.Physics, state: SimState) -> bool:
        qpos = physics.named.data.qpos
        original = qpos[self._joint][:2].copy()
        contacts_before = int(physics.data.ncon)
        for _ in range(NUDGE_ATTEMPTS):
            angle = float(self._rng.uniform(0.0, 2 * np.pi))
            shift = self._offset * np.array([np.cos(angle), np.sin(angle)])
            qpos[self._joint][:2] = original + shift
            physics.forward()
            # Any new contact would mean the object landed inside the gripper or off the table.
            if int(physics.data.ncon) <= contacts_before:
                physics.named.data.qvel[self._joint] = 0.0
                physics.forward()
                self.parameters.update(angle_rad=angle)
                return True
        qpos[self._joint][:2] = original
        physics.forward()
        return False


class PegMisalign(Perturbation):
    """Rotates the held peg between the fingers, about the axis the fingers close along."""

    def __init__(
        self, settings: PerturbationSettings, task: TaskSettings, rng: np.random.Generator
    ) -> None:
        """Samples a signed angle."""
        super().__init__(settings, task, rng)
        low, high = settings.angle_deg
        sign = 1.0 if rng.integers(0, 2) else -1.0
        self._angle = float(np.deg2rad(rng.uniform(low, high))) * sign
        self._joint = task.objects[self._target()].joint
        self._geoms = settings.finger_geoms
        self.parameters.update(angle_deg=float(np.rad2deg(self._angle)))

    def _window_open(self, state: SimState, phase: int) -> bool:
        return self._target_held(state, phase)

    def _fire(self, physics: mujoco.Physics, state: SimState) -> bool:
        data = physics.named.data
        first, second = (np.array(data.geom_xpos[geom]) for geom in self._geoms)
        axis = (second - first) / np.linalg.norm(second - first)
        half = self._angle / 2
        spin = np.concatenate([[np.cos(half)], np.sin(half) * axis])
        pose = np.array(data.qpos[self._joint])
        rotated = np.zeros(4)
        mujoco.mju_mulQuat(rotated, spin, pose[3:])
        # Rotate about the pinch point so that the peg stays between the fingers.
        arm = pose[:3] - (first + second) / 2
        along = axis * float(axis @ arm)
        moved = np.zeros(3)
        mujoco.mju_rotVecQuat(moved, arm - along, spin)
        data.qpos[self._joint] = np.concatenate([(first + second) / 2 + along + moved, rotated])
        physics.forward()
        return True


_CLASSES: dict[str, type[Perturbation]] = {
    "knock": Knock,
    "slip": Slip,
    "nudge": Nudge,
    "peg_misalign": PegMisalign,
}


def make_perturbation(
    name: str, task: TaskSettings, settings: Settings, seed: int
) -> Perturbation | None:
    """Builds the named perturbation, or None for nominal episodes.

    Args:
        name: One of `PERTURBATIONS`, or "none".
        task: Task the episode runs.
        settings: Global settings.
        seed: Episode seed; with the perturbation name it fixes all sampled parameters.

    Returns:
        A fresh perturbation for one episode.
    """
    if name == "none":
        return None
    rng = np.random.default_rng([seed, PERTURBATIONS.index(name)])
    return _CLASSES[name](settings.perturbations[name], task, rng)
