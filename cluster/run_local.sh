#!/usr/bin/env bash
# Run a sweep on one multi-GPU machine without Slurm: one shard per GPU per stage.
# Usage: bash cluster/run_local.sh configs/sweeps/pilot.yaml 0,1,2,3,4,5,6,7
# Repeat a GPU to run several processes on it (0,0,1,1). The scoring stages use each GPU once,
# because they measure latency; score_vlm starts one vLLM server per GPU (scripts/serve_vlm.sh,
# model from the spec's settings).
# EGL_PER_GPU=1 also renders on the shard's GPU (native Linux only, not WSL).
# RUN_DIR=runs/<name>_<stamp> continues an interrupted run: finished episodes and scores are kept.
# VLLM_EXTRA_ARGS adds vllm serve flags (server/check.sh records the ones this machine needs).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/vlm_lib.sh
source scripts/vlm_lib.sh

SPEC="$1"
IFS=',' read -r -a GPUS <<< "${2:-0}"
mapfile -t UNIQUE_GPUS < <(printf "%s\n" "${GPUS[@]}" | sort -un)
read -r NAME VLM_MODEL BASE_PORT < <(python -c '
import sys

import yaml

from amon.settings import load_settings

spec = yaml.safe_load(open(sys.argv[1]))
base = yaml.safe_load(open("configs/base.yaml"))
print(spec["name"], load_settings(overrides=spec.get("overrides")).vlm.model,
      base["server"]["vlm_base_port"])' "$SPEC")
RUN_DIR="${RUN_DIR:-runs/${NAME}_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR/logs"
[[ -f "$RUN_DIR/spec.yaml" ]] || cp "$SPEC" "$RUN_DIR/spec.yaml"
export MUJOCO_GL=egl
echo "[run_local] $SPEC -> $RUN_DIR"

PIDS=()
# Ctrl+C stops the shards and any vLLM servers (they run in their own process groups).
stop_everything() {
  echo "stopping"
  kill "${PIDS[@]}" 2> /dev/null
  for pid_file in "$RUN_DIR"/logs/vllm_shard*.pid; do
    [[ -f "$pid_file" ]] && kill -TERM -- "-$(cat "$pid_file")" 2> /dev/null
    rm -f "$pid_file"
  done
  exit 130
}
trap stop_everything INT TERM

run_stage_shard() {
  local stage="$1" shard="$2" gpu="$3" shards="$4"
  local log="$RUN_DIR/logs/shard_${shard}_${stage}.out"
  local endpoint_args=()
  local client_gpu="$gpu"
  local status=0
  if [[ "$stage" == "score_vlm" ]]; then
    local port=$((BASE_PORT + shard))
    # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS holds separate flags
    if ! start_vlm_server "$gpu" "$port" "$VLM_MODEL" "$RUN_DIR/logs/vllm_shard${shard}.log" \
      ${VLLM_EXTRA_ARGS:-}; then
      echo "shard=$shard stage=$stage status=1 (vLLM server did not start)" >> "$log"
      return 1
    fi
    echo "$VLM_PID" > "$RUN_DIR/logs/vllm_shard${shard}.pid"
    echo "vllm serve $VLM_MODEL ${VLLM_EXTRA_ARGS:-} on GPU $gpu" >> "$log"
    endpoint_args=(--endpoint "http://127.0.0.1:$port/v1")
    client_gpu=""  # the client needs no GPU; the server holds almost all of its memory
  fi
  [[ "${EGL_PER_GPU:-0}" == 1 ]] && export MUJOCO_EGL_DEVICE_ID="$gpu"
  CUDA_VISIBLE_DEVICES="$client_gpu" python -m experiments.run --spec "$SPEC" --stage "$stage" \
    --run-dir "$RUN_DIR" --shard "$shard" --of "$shards" "${endpoint_args[@]}" 2>&1 \
    | tee -a "$log" | sed "s/^/[shard $shard $stage] /" || status="${PIPESTATUS[0]}"
  stop_vlm_server
  rm -f "$RUN_DIR/logs/vllm_shard${shard}.pid"
  echo "shard=$shard stage=$stage status=$status" >> "$log"
  return "$status"
}

STAGES="$(python -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['stages']))" "$SPEC")"
for STAGE in $STAGES; do
  echo "[run_local] stage $STAGE started $(date +%H:%M:%S)"
  if [[ "$STAGE" == "train_wm" || "$STAGE" == "baseline_report" || "$STAGE" == "evaluate" \
    || "$STAGE" == "plot" ]]; then
    STAGE_GPUS=("${GPUS[0]}")
  elif [[ "$STAGE" == "score_vlm" || "$STAGE" == "score_wm" ]]; then
    # One shard per GPU: both stages time their monitor, and a shared GPU would inflate latency.
    STAGE_GPUS=("${UNIQUE_GPUS[@]}")
  else
    STAGE_GPUS=("${GPUS[@]}")
  fi
  PIDS=()
  for i in "${!STAGE_GPUS[@]}"; do
    run_stage_shard "$STAGE" "$i" "${STAGE_GPUS[$i]}" "${#STAGE_GPUS[@]}" &
    PIDS+=($!)
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
  # Later stages would silently run on incomplete data, so stop here; finished episodes are kept.
  if [[ "$FAILED" == 1 ]]; then
    echo "[run_local] stage $STAGE had failing shards; see $RUN_DIR/logs"
    echo "[run_local] after fixing the cause, continue with: RUN_DIR=$RUN_DIR and the same command"
    bash cluster/collect.sh "$RUN_DIR"
    exit 1
  fi
done

bash cluster/collect.sh "$RUN_DIR"
