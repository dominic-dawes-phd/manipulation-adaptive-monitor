"""Typed access to configs/base.yaml, the only home for global parameters.

Sections are added here in the milestone that first reads them.
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


@dataclass(frozen=True)
class ObjectSettings:
    """MuJoCo names and the resting center height of one task object."""

    body: str
    joint: str
    geoms: tuple[str, ...]
    rest_height_m: float


@dataclass(frozen=True)
class TaskSettings:
    """One gym-aloha task and the frozen checkpoint that runs it."""

    name: str
    env_id: str
    obs_type: str
    episode_len: int
    success_reward: int
    policy_repo: str
    policy_revision: str
    published_success_rate: float
    published_per_episode: bool
    objects: dict[str, ObjectSettings]


@dataclass(frozen=True)
class RobotSettings:
    """Action layout and gripper names."""

    action_dim: int
    groups: dict[str, tuple[int, ...]]
    gripper_close_threshold: float
    finger_geoms: dict[str, tuple[str, ...]]
    reward_finger_geoms: dict[str, str]


@dataclass(frozen=True)
class StatsSettings:
    """Resampling and interval parameters."""

    bootstrap_resamples: int
    ci: float


@dataclass(frozen=True)
class Settings:
    """All parameters used by the implemented milestones."""

    control_hz: int
    tasks: dict[str, TaskSettings]
    robot: RobotSettings
    splits: dict[str, tuple[int, int]]
    stats: StatsSettings


def load_settings(path: Path = BASE_CONFIG_PATH) -> Settings:
    """Loads and validates the base configuration.

    Args:
        path: YAML file to read.

    Returns:
        The typed settings.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    robot = raw["robot"]
    settings = Settings(
        control_hz=raw["project"]["control_hz"],
        tasks={name: _parse_task(name, task) for name, task in raw["tasks"].items()},
        robot=RobotSettings(
            action_dim=robot["action_dim"],
            groups={name: tuple(indices) for name, indices in robot["groups"].items()},
            gripper_close_threshold=robot["gripper_close_threshold"],
            finger_geoms={arm: tuple(geoms) for arm, geoms in robot["finger_geoms"].items()},
            reward_finger_geoms=dict(robot["reward_finger_geoms"]),
        ),
        splits={
            name: (bounds[0], bounds[1])
            for name, bounds in raw["splits"].items()
            if isinstance(bounds, list)
        },
        stats=StatsSettings(**raw["stats"]),
    )
    _validate(settings)
    return settings


def _parse_task(name: str, task: dict) -> TaskSettings:
    """Builds one task's settings from its YAML mapping."""
    objects = {
        object_name: ObjectSettings(
            body=spec["body"],
            joint=spec["joint"],
            geoms=tuple(spec["geoms"]),
            rest_height_m=spec["rest_height_m"],
        )
        for object_name, spec in task["objects"].items()
    }
    return TaskSettings(**{**task, "name": name, "objects": objects})


def _validate(settings: Settings) -> None:
    """Raises ValueError if the action groups or seed ranges are inconsistent."""
    covered = sorted(index for group in settings.robot.groups.values() for index in group)
    if covered != list(range(settings.robot.action_dim)):
        raise ValueError(f"robot.groups must cover every action index exactly once: {covered}")
    for (_, earlier_end), (later_start, _) in pairwise(sorted(settings.splits.values())):
        if later_start < earlier_end:
            raise ValueError(f"split seed ranges overlap: {settings.splits}")
