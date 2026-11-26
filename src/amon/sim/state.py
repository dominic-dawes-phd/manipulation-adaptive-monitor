"""Reads the simulator state that labels and perturbations are defined on.

Everything here is privileged. Schedules and triggers must never import this module.
"""

from dataclasses import dataclass

import numpy as np
from dm_control import mujoco

from amon.settings import Settings, TaskSettings

ARMS = ("left", "right")
LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class SimState:
    """Privileged state at one control step, for K task objects and two arms."""

    obj_pos: np.ndarray  # [K, 3] meters
    contact: np.ndarray  # [K, 2] bool, any finger of the arm touches the object
    held: np.ndarray  # [K, 2] bool, contact and lifted above the resting height
    gripper_pos: np.ndarray  # [2, 3] meters, midpoint of the arm's two finger geoms


class StateReader:
    """Resolves MuJoCo names once and reads `SimState` from the physics."""

    def __init__(self, physics: mujoco.Physics, task: TaskSettings, settings: Settings) -> None:
        """Looks up the geom ids of the task objects and the fingers."""
        model = physics.model
        self._bodies = [spec.body for spec in task.objects.values()]
        self._rest_heights = np.array([spec.rest_height_m for spec in task.objects.values()])
        self._hold_margin = settings.labels.hold_margin_m
        self._object_geoms = [
            np.array([model.name2id(geom, "geom") for geom in spec.geoms])
            for spec in task.objects.values()
        ]
        self._finger_geoms = [
            np.array([model.name2id(geom, "geom") for geom in settings.robot.finger_geoms[arm]])
            for arm in ARMS
        ]

    def read(self, physics: mujoco.Physics) -> SimState:
        """Returns the current state."""
        obj_pos = np.array([physics.named.data.xpos[body] for body in self._bodies])
        contacts = physics.data.contact[: physics.data.ncon]
        first = np.array([contact.geom1 for contact in contacts], dtype=int)
        second = np.array([contact.geom2 for contact in contacts], dtype=int)
        touching = np.zeros((len(self._bodies), len(ARMS)), dtype=bool)
        for k, geoms in enumerate(self._object_geoms):
            for arm, fingers in enumerate(self._finger_geoms):
                forward = np.isin(first, geoms) & np.isin(second, fingers)
                backward = np.isin(first, fingers) & np.isin(second, geoms)
                touching[k, arm] = bool(np.any(forward | backward))
        lifted = obj_pos[:, 2] > self._rest_heights + self._hold_margin
        return SimState(
            obj_pos=obj_pos,
            contact=touching,
            held=touching & lifted[:, None],
            gripper_pos=np.array(
                [physics.data.geom_xpos[fingers].mean(axis=0) for fingers in self._finger_geoms]
            ),
        )
