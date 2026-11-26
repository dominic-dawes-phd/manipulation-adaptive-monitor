#!/usr/bin/env bash
# Tests + lint. Run before reporting any milestone done.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python -m pytest -q
ruff check src experiments tests scripts
ruff format --check src experiments tests scripts
