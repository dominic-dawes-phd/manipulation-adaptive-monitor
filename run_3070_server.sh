#!/usr/bin/env bash
# Run a sweep on the 8x RTX 3070 server (no Slurm). Step-by-step guide: docs/server.md.
#
#   bash server/setup.sh                                      # once: environments and weights
#   bash run_3070_server.sh --check                           # before any data, after updates
#   bash run_3070_server.sh configs/sweeps/smoke.yaml         # a sweep on all GPUs
#   bash run_3070_server.sh configs/sweeps/pilot.yaml 0,1,2,3 # some GPUs
#
# Settings (environment variables):
#   AMON_HOME        environments and caches from server/setup.sh, default ~/amon-3070
#   PROCS_PER_GPU    rollout and world-model processes per GPU, default 2
#   RUN_DIR          continue an interrupted run, e.g. RUN_DIR=runs/pilot_20261127-101500
#   ALLOW_OTHER_GPU  set to 1 to run on a GPU that is not an RTX 3070
#
# Results from this machine must not be mixed with laptop results in one comparison (D-09).
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=server/common.sh
source server/common.sh

if [[ "${1:-}" == "--check" ]]; then
  shift
  exec bash server/check.sh "$@"
fi
SPEC="${1:-configs/sweeps/pilot.yaml}"
[[ -f "$SPEC" ]] || fail "no such sweep spec: $SPEC"
load_env
select_gpus "${2:-}"
if [[ "$(cat "$AMON_HOME/check.pass" 2> /dev/null)" != "$(check_fingerprint)" ]]; then
  fail "the server check has not passed for this commit: run 'bash run_3070_server.sh --check'"
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[server] WARNING: uncommitted changes to tracked files; run.json will mark the run dirty"
fi

# Several processes share a GPU: one ACT process uses about 0.5 GB and is CPU-bound.
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
SHARD_GPUS=""
for _ in $(seq 1 "$PROCS_PER_GPU"); do
  SHARD_GPUS="${SHARD_GPUS:+$SHARD_GPUS,}$GPU_LIST"
done
export EGL_PER_GPU="${EGL_PER_GPU:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
echo "[server] sweep $SPEC on GPUs $GPU_LIST, $PROCS_PER_GPU processes per GPU"
exec bash cluster/run_local.sh "$SPEC" "$SHARD_GPUS"
