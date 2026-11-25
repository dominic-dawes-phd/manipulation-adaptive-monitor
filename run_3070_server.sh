#!/usr/bin/env bash
# One command to run a sweep on the 8x RTX 3070 server (no Slurm): installs the locked
# environment on first use, checks the machine, then runs every stage on every GPU.
#
#   bash run_3070_server.sh                                   # pilot sweep, all GPUs
#   bash run_3070_server.sh configs/sweeps/main.yaml          # another sweep
#   bash run_3070_server.sh configs/sweeps/main.yaml 0,1,2,3  # some GPUs
#   bash run_3070_server.sh --check                           # install and check only
#
# Settings (environment variables):
#   AMON_HOME        environment and caches, default ~/amon-3070 (needs about 15 GB)
#   PROCS_PER_GPU    rollout processes per GPU, default 2
#   ALLOW_OTHER_GPU  set to 1 to run on a GPU that is not an RTX 3070
#   SKIP_TESTS       set to 1 to skip the determinism tests (about 2 minutes)
#
# Results from this machine must not be mixed with laptop results in one comparison (D-09).
set -euo pipefail
cd "$(dirname "$0")"

AMON_HOME="${AMON_HOME:-$HOME/amon-3070}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
REQUIREMENTS="requirements/server_3070.txt"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi
SPEC="${1:-configs/sweeps/pilot.yaml}"
[[ -f "$SPEC" ]] || { echo "[server] no such sweep spec: $SPEC"; exit 1; }

# GPUs: same numbering for CUDA, nvidia-smi and the per-GPU renderer.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
command -v nvidia-smi >/dev/null || { echo "[server] nvidia-smi not found"; exit 1; }
GPU_LIST="${2:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)"
  echo "[server] GPU $gpu: $name"
  if [[ "$name" != *"RTX 3070"* && "${ALLOW_OTHER_GPU:-0}" != 1 ]]; then
    echo "[server] expected an RTX 3070. If this is intended: ALLOW_OTHER_GPU=1 bash $0 ..."
    exit 1
  fi
done

# Environment: locked versions, rebuilt only when the lock file changes.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AMON_HOME/uv-cache}" HF_HOME="${HF_HOME:-$AMON_HOME/hf}"
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$AMON_HOME"
command -v uv >/dev/null || python3 -m pip install --user uv
STAMP_FILE="$AMON_HOME/installed.sha256"
WANTED="$(sha256sum "$REQUIREMENTS" pyproject.toml)"
if [[ ! -x "$AMON_HOME/venv/bin/python" || "$(cat "$STAMP_FILE" 2>/dev/null)" != "$WANTED" ]]; then
  echo "[server] installing into $AMON_HOME/venv"
  [[ -x "$AMON_HOME/venv/bin/python" ]] || uv venv "$AMON_HOME/venv" --python 3.11
  uv pip sync -p "$AMON_HOME/venv/bin/python" "$REQUIREMENTS"
  uv pip install -p "$AMON_HOME/venv/bin/python" --no-deps -e .
  echo "$WANTED" > "$STAMP_FILE"
fi
# shellcheck disable=SC1091
source "$AMON_HOME/venv/bin/activate"

# Determinism depends on this machine's renderer and driver, so it is re-checked here (D-12).
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
  echo "[server] running the test suite on GPU ${GPUS[0]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m pytest -q -W ignore || {
    echo "[server] tests failed; do not produce results on this machine until they pass"
    exit 1
  }
fi
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "[server] ready"
  exit 0
fi

# Several rollout processes share a GPU: one ACT process uses about 0.5 GB and is CPU-bound.
SHARD_GPUS=""
for _ in $(seq 1 "$PROCS_PER_GPU"); do
  SHARD_GPUS="${SHARD_GPUS:+$SHARD_GPUS,}$GPU_LIST"
done
export EGL_PER_GPU=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
echo "[server] sweep $SPEC on GPUs $GPU_LIST, $PROCS_PER_GPU processes per GPU"
exec bash cluster/run_local.sh "$SPEC" "$SHARD_GPUS"
