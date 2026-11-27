# Shared by server/setup.sh, server/check.sh and run_3070_server.sh. Source it; do not run it.
#
# Environment: AMON_HOME (default ~/amon-3070) holds the environments, model caches and env.sh.
# ALLOW_OTHER_GPU=1 accepts GPUs that are not RTX 3070s (for testing the scripts elsewhere).

AMON_HOME="${AMON_HOME:-$HOME/amon-3070}"
export AMON_HOME
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # same GPU numbering for CUDA, nvidia-smi and EGL

fail() {
  echo "[amon] ERROR: $*" >&2
  exit 1
}

# Loads env.sh from setup and activates the simulation environment.
load_env() {
  [[ -f "$AMON_HOME/env.sh" ]] || fail "no $AMON_HOME/env.sh: run 'bash server/setup.sh' first"
  # The stamps are sha256sum output for the files each environment was installed from.
  sha256sum --check --status "$AMON_HOME/sim.stamp" 2> /dev/null \
    || fail "the simulation environment is older than the code: rerun 'bash server/setup.sh'"
  if [[ -f "$AMON_HOME/vlm.stamp" ]]; then
    sha256sum --check --status "$AMON_HOME/vlm.stamp" 2> /dev/null \
      || fail "the VLM environment is older than the code: rerun 'bash server/setup.sh'"
  fi
  # shellcheck disable=SC1091
  source "$AMON_HOME/env.sh"
  # shellcheck disable=SC1091
  source "$SIM_VENV/bin/activate"
}

# Sets the array GPUS from a comma-separated list (default: every GPU) and checks their model.
select_gpus() {
  command -v nvidia-smi > /dev/null || fail "nvidia-smi not found; is the NVIDIA driver installed?"
  local list="${1:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
  IFS=',' read -r -a GPUS <<< "$list"
  local gpu name
  for gpu in $(printf "%s\n" "${GPUS[@]}" | sort -un); do
    name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" \
      || fail "GPU $gpu does not exist"
    echo "[amon] GPU $gpu: $name"
    if [[ "$name" != *"RTX 3070"* && "${ALLOW_OTHER_GPU:-0}" != 1 ]]; then
      fail "GPU $gpu is not an RTX 3070. To use it anyway: ALLOW_OTHER_GPU=1 (never mix results, D-09)"
    fi
  done
}

# Prints the fingerprint a passed server check is valid for: code commit and lock files.
check_fingerprint() {
  echo "$(git rev-parse HEAD) $(cat requirements/server_3070.txt requirements/vlm_*.txt | sha256sum | cut -c1-16)"
}
