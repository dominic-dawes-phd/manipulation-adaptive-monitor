# Start and stop a vLLM server for one GPU. Source it from cluster/run_local.sh and
# server/check.sh; do not run it. The server runs in its own process group, so stopping it also
# stops vLLM's engine processes.

# start_vlm_server GPU PORT MODEL LOG [extra vllm serve flags...]
# Returns 0 once /health answers; 1 if the server exits or does not answer within
# server.vlm_health_timeout_s (configs/base.yaml). Sets VLM_PID either way.
start_vlm_server() {
  local gpu="$1" port="$2" model="$3" log="$4"
  shift 4
  local timeout
  timeout="$(python -c 'import yaml; print(yaml.safe_load(open("configs/base.yaml"))["server"]["vlm_health_timeout_s"])')"
  CUDA_VISIBLE_DEVICES="$gpu" VLM_PORT="$port" setsid bash scripts/serve_vlm.sh "$model" "$@" \
    > "$log" 2>&1 &
  VLM_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    if ! kill -0 "$VLM_PID" 2> /dev/null; then
      echo "[vlm] server for $model on GPU $gpu exited; last lines of $log:" >&2
      tail -n 15 "$log" >&2
      return 1
    fi
    if (( waited >= timeout )); then
      echo "[vlm] server for $model on GPU $gpu not ready after ${timeout}s; see $log" >&2
      stop_vlm_server
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "[vlm] $model ready on GPU $gpu, port $port, after ${waited}s"
}

# Stops the server started by start_vlm_server, if it is still running.
stop_vlm_server() {
  [[ -n "${VLM_PID:-}" ]] || return 0
  kill -TERM -- "-$VLM_PID" 2> /dev/null || true
  local waited=0
  while kill -0 "$VLM_PID" 2> /dev/null && (( waited < 30 )); do
    sleep 1
    waited=$((waited + 1))
  done
  kill -KILL -- "-$VLM_PID" 2> /dev/null || true
  wait "$VLM_PID" 2> /dev/null || true
  VLM_PID=""
}
