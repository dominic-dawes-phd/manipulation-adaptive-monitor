#!/usr/bin/env bash
# Checks the server before it produces any data (D-12, D-22). Run after server/setup.sh and again
# after every code update; sweeps refuse to start until it has passed for the current commit.
#
#   bash run_3070_server.sh --check [GPU list]      (same as: bash server/check.sh [GPU list])
#
# 1. The test suite, including the in-process determinism tests.
# 2. Determinism across processes and GPUs: configs/sweeps/server_check.yaml is rolled out on the
#    first two GPUs and every stored array is compared bit for bit.
# 3. The VLM: does the 7B model load on one GPU (then with --enforce-eager, then the 3B), and how
#    long does one request take. Scores the second copy of the check episodes.
#
# Writes runs/server_check_<stamp>/ with check_summary.txt. Environment: VLLM_EXTRA_ARGS (extra
# vllm serve flags for every attempt), EGL_PER_GPU (default 1; 0 under WSL), SKIP_VLM=1.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=server/common.sh
source server/common.sh
# shellcheck source=scripts/vlm_lib.sh
source scripts/vlm_lib.sh
USER_VLLM_ARGS="${VLLM_EXTRA_ARGS:-}"
rm -f "$AMON_HOME/vlm.env" "$AMON_HOME/check.pass"
load_env
select_gpus "${1:-}"
FIRST="${GPUS[0]}"
SECOND="${GPUS[1]:-${GPUS[0]}}"
RUN_DIR="runs/server_check_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
TESTS=FAIL
DETERMINISM=FAIL
VLM=FAIL
VLM_NOTE="not run"
trap 'stop_vlm_server' EXIT

echo "== 1/3 test suite on GPU $FIRST (about 3 minutes) =="
if CUDA_VISIBLE_DEVICES="$FIRST" python -m pytest -q -W ignore 2>&1 | tee "$RUN_DIR/pytest.log"; then
  TESTS=PASS
fi

echo "== 2/3 determinism: the same episodes on GPU $FIRST and GPU $SECOND (about 5 minutes) =="
COPIES=("$RUN_DIR/a_gpu$FIRST" "$RUN_DIR/b_gpu$SECOND")
PIDS=()
for index in 0 1; do
  gpu="$FIRST"
  (( index == 1 )) && gpu="$SECOND"
  egl_device=()
  [[ "${EGL_PER_GPU:-1}" == 1 ]] && egl_device=(MUJOCO_EGL_DEVICE_ID="$gpu")
  env CUDA_VISIBLE_DEVICES="$gpu" "${egl_device[@]}" python -m experiments.run \
    --spec configs/sweeps/server_check.yaml --stage rollout --run-dir "${COPIES[$index]}" \
    > "$RUN_DIR/rollout_$index.log" 2>&1 &
  PIDS+=($!)
done
ROLLOUTS_OK=1
for pid in "${PIDS[@]}"; do wait "$pid" || ROLLOUTS_OK=0; done
if (( ROLLOUTS_OK )) && python scripts/compare_runs.py "${COPIES[@]}" 2>&1 | tee "$RUN_DIR/compare.log"; then
  DETERMINISM=PASS
elif (( ! ROLLOUTS_OK )); then
  echo "rollouts failed; see $RUN_DIR/rollout_0.log and rollout_1.log"
fi

if [[ "${SKIP_VLM:-0}" == 1 ]]; then
  VLM="SKIPPED"
else
  echo "== 3/3 VLM on GPU $FIRST (5-15 minutes; the first start compiles kernels) =="
  read -r MAIN_MODEL FALLBACK_MODEL PORT < <(python -c '
import yaml
config = yaml.safe_load(open("configs/base.yaml"))
print(config["monitors"]["vlm"]["model"], config["monitors"]["vlm"]["fallback_model"],
      config["server"]["vlm_base_port"])')
  ATTEMPTS=("$MAIN_MODEL|" "$MAIN_MODEL|--enforce-eager" "$FALLBACK_MODEL|")
  for number in "${!ATTEMPTS[@]}"; do
    attempt="${ATTEMPTS[$number]}"
    model="${attempt%%|*}"
    flags="$(echo "$USER_VLLM_ARGS ${attempt#*|}" | xargs)"
    echo "attempt $((number + 1)): $model ${flags:+with $flags}"
    # shellcheck disable=SC2086  # flags are separate words
    if ! start_vlm_server "$FIRST" "$PORT" "$model" "$RUN_DIR/vllm_attempt$((number + 1)).log" $flags; then
      continue
    fi
    python - "$RUN_DIR/vlm_spec.yaml" "$model" << 'EOF'
import sys

import yaml

spec = yaml.safe_load(open("configs/sweeps/server_check.yaml"))
spec["overrides"]["monitors.vlm.model"] = sys.argv[2]
spec["stages"] = ["score_vlm"]
yaml.safe_dump(spec, open(sys.argv[1], "w"))
EOF
    if CUDA_VISIBLE_DEVICES="" python -m experiments.run --spec "$RUN_DIR/vlm_spec.yaml" \
      --stage score_vlm --run-dir "${COPIES[1]}" --endpoint "http://127.0.0.1:$PORT/v1" \
      2>&1 | tee "$RUN_DIR/score_vlm.log"; then
      VLM=PASS
      VLM_NOTE="$model ${flags:+($flags) }$(python -c '
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("scores/vlm/meta_*_shard0.json")):
    meta = json.loads(path.read_text())
    latency = meta["latency_ms"]
    size = meta["frame_size"][0]
    print("%s %d px: %.0f ms mean, %.0f ms p95;" % (path.stem[5:].rsplit("_shard", 1)[0], size, latency["mean"], latency["p95"]), end=" ")
' "${COPIES[1]}")"
      stop_vlm_server
      break
    fi
    stop_vlm_server
  done
  if [[ "$VLM" == PASS && -n "$flags" ]]; then
    echo "export VLLM_EXTRA_ARGS=\"$flags\"" > "$AMON_HOME/vlm.env"
  fi
fi

{
  echo "server check $(date -Is) on $(hostname), commit $(git rev-parse --short HEAD)"
  echo "  tests        $TESTS   ($RUN_DIR/pytest.log)"
  echo "  determinism  $DETERMINISM   ($RUN_DIR/compare.log)"
  echo "  vlm          $VLM   $VLM_NOTE"
} | tee "$RUN_DIR/check_summary.txt"

if [[ "$VLM" == PASS && "$VLM_NOTE" != "$MAIN_MODEL "* ]]; then
  echo
  echo "The main VLM model did not load. Add this line under 'overrides:' in"
  echo "configs/sweeps/smoke.yaml and configs/sweeps/pilot.yaml before running them:"
  echo "  monitors.vlm.model: $FALLBACK_MODEL"
fi
if [[ "$TESTS" == PASS && "$DETERMINISM" == PASS ]]; then
  check_fingerprint > "$AMON_HOME/check.pass"
  echo
  echo "Server check passed. Next: bash run_3070_server.sh configs/sweeps/smoke.yaml"
else
  echo
  echo "Server check FAILED. Do not produce results on this machine. Send back $RUN_DIR/ and"
  echo "$AMON_HOME/server_info.txt."
  exit 1
fi
