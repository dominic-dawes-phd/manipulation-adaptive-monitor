#!/usr/bin/env bash
# Run a sweep on one multi-GPU machine without Slurm: one shard per GPU per stage.
# Usage: bash cluster/run_local.sh configs/sweeps/pilot.yaml 0,1,2,3,4,5,6,7
set -euo pipefail
cd "$(dirname "$0")/.."

SPEC="$1"
IFS=',' read -r -a GPUS <<< "${2:-0}"
NUM_SHARDS="${#GPUS[@]}"
NAME="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$SPEC")"
RUN_DIR="runs/${NAME}_$(date +%Y%m%d-%H%M%S)"
VLLM_BIN="${VLLM_BIN:-.venv-vlm/bin/vllm}"
mkdir -p "$RUN_DIR/logs"
cp "$SPEC" "$RUN_DIR/spec.yaml"
export MUJOCO_GL=egl

PIDS=()
trap 'echo "stopping"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT

run_stage_shard() {
  local stage="$1" shard="$2" gpu="$3" shards="$4"
  local log="$RUN_DIR/logs/shard_${shard}_${stage}.out"
  local endpoint_args=()
  local vllm_pid=""
  if [[ "$stage" == "score_vlm" ]]; then
    local port=$((8100 + shard))
    CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_BIN" serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
      --port "$port" --max-model-len 4096 --gpu-memory-utilization 0.92 \
      --limit-mm-per-prompt '{"image": 1}' > "$RUN_DIR/logs/vllm_shard${shard}.log" 2>&1 &
    vllm_pid=$!
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do sleep 5; done
    endpoint_args=(--endpoint "http://127.0.0.1:$port/v1")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -m experiments.run --spec "$SPEC" --stage "$stage" \
    --run-dir "$RUN_DIR" --shard "$shard" --of "$shards" "${endpoint_args[@]}" 2>&1 \
    | tee "$log" | sed "s/^/[shard $shard $stage] /"
  local status="${PIPESTATUS[0]}"
  [[ -n "$vllm_pid" ]] && kill "$vllm_pid" 2>/dev/null
  return "$status"
}

STAGES="$(python -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['stages']))" "$SPEC")"
for STAGE in $STAGES; do
  if [[ "$STAGE" == "train_wm" || "$STAGE" == "evaluate" || "$STAGE" == "plot" ]]; then
    run_stage_shard "$STAGE" 0 "${GPUS[0]}" 1
    continue
  fi
  PIDS=()
  for i in "${!GPUS[@]}"; do
    run_stage_shard "$STAGE" "$i" "${GPUS[$i]}" "$NUM_SHARDS" &
    PIDS+=($!)
  done
  FAILED=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
  [[ "$FAILED" == 1 ]] && echo "[run_local] stage $STAGE had failing shards; see $RUN_DIR/logs"
done

bash cluster/collect.sh "$RUN_DIR"
