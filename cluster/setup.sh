#!/usr/bin/env bash
# One-time setup. Run INSIDE a GPU job (srun ... --pty bash cluster/setup.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this inside a GPU job, for example:"
  echo "  srun $SLURM_CLUSTER_FLAG --account=$ACCOUNT --gres=gpu:1 --time=01:00:00 --pty bash cluster/setup.sh"
  exit 1
fi

mkdir -p "$DATA_DIR" "$HF_HOME" "$RUNS_DIR"
export HF_HOME UV_CACHE_DIR="$DATA_DIR/uv-cache" PIP_CACHE_DIR="$DATA_DIR/pip-cache"

command -v uv >/dev/null || pip install --user uv

uv venv "$VENV_SIM" --python 3.11
VIRTUAL_ENV="$VENV_SIM" uv pip install -e ".[dev]"

uv venv "$VENV_VLM" --python 3.11
VIRTUAL_ENV="$VENV_VLM" uv pip install vllm

source "$VENV_SIM/bin/activate"
bash scripts/fetch_models.sh "$HF_HOME"

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  sinfo $SLURM_CLUSTER_FLAG -s || true
  sacctmgr -n show assoc user="$USER" format=account,partition,qos || true
  df -h "$DATA_DIR"
} > cluster_info.txt
echo "Setup done. Send cluster_info.txt back if anything looks off."
