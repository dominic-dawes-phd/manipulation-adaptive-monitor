#!/usr/bin/env bash
# Starts the vLLM server for the VLM judge (spec section 6.1) from the separate VLM environment
# (D-08). Model, pixel cap, logprob count, port and serving limits come from monitors.vlm in
# configs/base.yaml.
#
#   scripts/serve_vlm.sh                 # monitors.vlm.model
#   scripts/serve_vlm.sh <model> [extra vllm serve flags]
#
# Environment: VLM_VENV (default ~/amon/venv-vlm); VLM_PORT (default: the endpoint's port); CC
# (Triton compiles kernel launchers with a C compiler; without root, server/setup.sh installs
# `ziglang` and a wrapper script that runs `python -m ziglang cc "$@"`).
set -euo pipefail
cd "$(dirname "$0")/.."
source "${VLM_VENV:-$HOME/amon/venv-vlm}/bin/activate"
if [[ -z "${CC:-}" ]] && ! command -v cc > /dev/null; then
  echo "no C compiler: install gcc or set CC (see the header of this script)" >&2
  exit 1
fi
read -r DEFAULT_MODEL MAX_PIXELS TOP_LOGPROBS PORT MAX_MODEL_LEN MAX_NUM_SEQS GPU_UTIL < <(python -c '
import urllib.parse
import yaml
vlm = yaml.safe_load(open("configs/base.yaml"))["monitors"]["vlm"]
port = urllib.parse.urlparse(vlm["endpoint"]).port
print(vlm["model"], vlm["max_pixels"], vlm["top_logprobs"], port, vlm["max_model_len"],
      vlm["max_num_seqs"], vlm["gpu_memory_utilization"])
')
PORT="${VLM_PORT:-$PORT}"
MODEL="${1:-$DEFAULT_MODEL}"
shift $(( $# > 0 ? 1 : 0 ))
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
  --max-logprobs "$TOP_LOGPROBS" \
  "$@"
