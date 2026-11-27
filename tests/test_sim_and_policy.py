"""Simulator names, determinism, and the shadow-query guarantee (CLAUDE.md rule 3)."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from amon.policy.act import ActAdapter, enable_determinism
from amon.rollout.runner import EpisodeTrace, run_episode
from amon.settings import load_settings
from amon.sim.env import AlohaSim
from amon.sim.state import StateReader

SETTINGS = load_settings()
SEED = 1001
RENDER_REPEATS = 8  # the multisampling flicker changed 1-2 pixels in about half of the renders
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module", params=sorted(SETTINGS.tasks))
def task_name(request: pytest.FixtureRequest) -> str:
    """Runs each test on every task."""
    return request.param


@pytest.fixture(scope="module")
def sim(task_name: str) -> Iterator[AlohaSim]:
    """Provides one environment per task."""
    environment = AlohaSim(SETTINGS.tasks[task_name], SETTINGS.render)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def policy(task_name: str) -> ActAdapter:
    """Provides the frozen policy of the task."""
    enable_determinism()
    return ActAdapter(SETTINGS.tasks[task_name])


@pytest.fixture(scope="module")
def traces(task_name: str, sim: AlohaSim, policy: ActAdapter) -> tuple[EpisodeTrace, EpisodeTrace]:
    """Runs the same seed with shadow queries off and on."""
    reader = StateReader(sim.physics, SETTINGS.tasks[task_name], SETTINGS)
    size = SETTINGS.render.store_size
    return (
        run_episode(sim, policy, reader, SEED, size, shadow_queries=False),
        run_episode(sim, policy, reader, SEED, size, shadow_queries=True),
    )


def test_configured_names_exist(task_name: str, sim: AlohaSim) -> None:
    """Every MuJoCo name in base.yaml exists in the model."""
    model = sim.physics.model
    for spec in SETTINGS.tasks[task_name].objects.values():
        assert model.name2id(spec.body, "body") >= 0
        assert model.name2id(spec.joint, "joint") >= 0
        for geom in spec.geoms:
            assert model.name2id(geom, "geom") >= 0
    for geoms in SETTINGS.robot.finger_geoms.values():
        for geom in geoms:
            assert model.name2id(geom, "geom") >= 0
    for arm, geom in SETTINGS.robot.reward_finger_geoms.items():
        assert geom in SETTINGS.robot.finger_geoms[arm]


def test_object_rest_heights(task_name: str, sim: AlohaSim) -> None:
    """Objects settle at the configured center heights when the robot holds still."""
    observation = sim.reset(SEED)
    for _ in range(40):
        sim.step(observation.agent_pos)
    for spec in SETTINGS.tasks[task_name].objects.values():
        height = float(sim.physics.named.data.xpos[spec.body][2])
        assert height == pytest.approx(spec.rest_height_m, abs=5e-4)


def test_reset_is_deterministic(sim: AlohaSim) -> None:
    """The same seed gives bit-identical pixels and state, which the replay design needs."""
    first = sim.reset(SEED)
    second = sim.reset(SEED)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.agent_pos, second.agent_pos)


def test_rendering_the_same_state_is_deterministic(sim: AlohaSim) -> None:
    """Repeated renders of one state are bit-identical (multisampling breaks this, D-29)."""
    sim.reset(SEED)
    assert sim.physics.model.vis.quality.offsamples == SETTINGS.render.offsamples
    first = sim.render("top")
    for _ in range(RENDER_REPEATS):
        assert np.array_equal(sim.render("top"), first)


def test_settings_match_checkpoint_eval_config(task_name: str) -> None:
    """Episode length and observation type equal the checkpoint's own evaluation settings."""
    task = SETTINGS.tasks[task_name]
    path = hf_hub_download(task.policy_repo, "train_config.json", revision=task.policy_revision)
    env = json.loads(Path(path).read_text(encoding="utf-8"))["env"]
    assert env["episode_length"] == task.episode_len
    assert env["obs_type"] == task.obs_type


@needs_cuda
def test_shadow_queries_do_not_change_execution(
    traces: tuple[EpisodeTrace, EpisodeTrace],
) -> None:
    """Executed actions, states and rewards are bit-identical with shadow queries on and off."""
    off, on = traces
    assert np.array_equal(off.action_exec, on.action_exec)
    assert np.array_equal(off.agent_pos, on.agent_pos)
    assert np.array_equal(off.reward, on.reward)
    assert off.success == on.success


@needs_cuda
def test_shadow_chunk_equals_executed_chunk_at_requery(
    policy: ActAdapter, traces: tuple[EpisodeTrace, EpisodeTrace]
) -> None:
    """Where the executor re-queries, the logged chunk is exactly the chunk it executes."""
    assert policy.temporal_ensemble_coeff is None
    _, trace = traces
    steps = len(trace.reward)
    for start in range(0, steps, policy.n_action_steps):
        length = min(policy.n_action_steps, steps - start)
        executed = trace.action_exec[start : start + length]
        assert np.array_equal(trace.chunks[start, :length], executed)


@needs_cuda
def test_shadow_query_is_deterministic_and_leaves_rng_alone(
    sim: AlohaSim, policy: ActAdapter
) -> None:
    """Repeated queries agree exactly and consume no random numbers."""
    observation = sim.reset(SEED)
    policy.reset()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    first = policy.shadow_chunk(observation)
    second = policy.shadow_chunk(observation)
    assert first.shape == (policy.chunk_size, SETTINGS.robot.action_dim)
    assert np.array_equal(first, second)
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(cuda_state, torch.cuda.get_rng_state())
