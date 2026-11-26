#!/usr/bin/env bash
# One shard of one stage. Submitted as an array by cluster/submit.sh.
# Expects env vars: SPEC, STAGE, STAMP, NUM_SHARDS, REPO_DIR.
#SBATCH --job-name=amon
#SBATCH --gres=gpu:1
#SBATCH --requeue
set -euo pipefail
cd "$REPO_DIR"
source cluster/cluster.env
export HF_HOME MUJOCO_GL=egl

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="$RUNS_DIR/$STAMP"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# GPU utilization log, one line a minute.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
  > "$LOG_DIR/gpu_shard${SHARD}_${STAGE}.csv" &
GPU_LOG_PID=$!

ENDPOINT_ARGS=()
if [[ "$STAGE" == "score_vlm" ]]; then
  PORT=$((VLLM_BASE_PORT + SHARD))
  "$VENV_VLM/bin/vllm" serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --port "$PORT" --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 1}' \
    > "$LOG_DIR/vllm_shard${SHARD}.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 5
  done
  ENDPOINT_ARGS=(--endpoint "http://127.0.0.1:$PORT/v1")
fi

source "$VENV_SIM/bin/activate"
set +e
python -m experiments.run --spec "$SPEC" --stage "$STAGE" \
  --run-dir "$RUN_DIR" --shard "$SHARD" --of "$NUM_SHARDS" "${ENDPOINT_ARGS[@]}"
STATUS=$?
set -e

[[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
kill "$GPU_LOG_PID" 2>/dev/null || true
echo "shard=$SHARD stage=$STAGE status=$STATUS restarts=${SLURM_RESTART_COUNT:-0}"
exit "$STATUS"
