"""run.json: everything needed to trace a result back to code, hardware and weights."""

import json
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path

import mujoco
import torch

from amon.rollout.runner import FRAME_RESIZE
from amon.settings import REPO_ROOT, Settings
from amon.version import AMON_VERSION

TRACKED_PACKAGES = (
    "lerobot",
    "gym-aloha",
    "gymnasium",
    "mujoco",
    "dm-control",
    "torch",
    "torchvision",
    "numpy",
)


def write_manifest(run_dir: Path, spec: dict, settings: Settings) -> None:
    """Writes run.json once per run; later shards and stages leave it unchanged."""
    path = run_dir / "run.json"
    if path.exists():
        return
    manifest = {
        "amon_version": AMON_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "gl_renderer": gl_renderer(),
        "packages": {name: version(name) for name in TRACKED_PACKAGES},
        "frame_resize": FRAME_RESIZE,
        "models": {task.policy_repo: task.policy_revision for task in settings.tasks.values()},
        "spec": spec,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def gl_renderer() -> str:
    """Names the OpenGL device MuJoCo renders on; the rendered pixels depend on it."""
    context = mujoco.GLContext(64, 64)
    context.make_current()
    # PyOpenGL must be imported after MuJoCo has selected its platform.
    from OpenGL import GL

    name = GL.glGetString(GL.GL_RENDERER).decode()
    context.free()
    return name


def _git(*arguments: str) -> str:
    """Runs a read-only git command in the repository and returns its output."""
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()
