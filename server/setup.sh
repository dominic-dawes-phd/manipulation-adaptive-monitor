#!/usr/bin/env bash
# One-time setup of the 8x RTX 3070 server. Needs internet, no sudo. Safe to rerun: finished
# steps are skipped, and a changed lock file reinstalls only its environment.
#
#   bash server/setup.sh
#
# Installs into AMON_HOME (default ~/amon-3070, about 50 GB):
#   venv/        simulation, policy and world model   (requirements/server_3070.txt)
#   venv-vlm/    vLLM server, separate torch (D-08)   (requirements/vlm_freeze.txt or vlm_cu12.txt)
#   hf/, torch/  model weights                        (scripts/fetch_models.py)
#   env.sh       sourced by every other script
#   server_info.txt   machine description; send it back if anything fails
#
# Environment: AMON_HOME, ALLOW_OTHER_GPU=1 (testing elsewhere), SKIP_VLM=1 (no VLM environment
# and no VLM weights), VLM_LOCK (force a VLM lock file). HF_HOME, TORCH_HOME and UV_CACHE_DIR are
# kept if already set, otherwise they go under AMON_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=server/common.sh
source server/common.sh

UV_VERSION="0.13.10"
MIN_DRIVER=525        # CUDA 12 minor-version compatibility; torch 2.7.1 cu126 and vLLM 0.11 cu128
CUDA13_DRIVER=580     # vLLM 0.34 ships torch for CUDA 13.0
MIN_GLIBC="2.31"
WANT_DISK_GB=50

mkdir -p "$AMON_HOME/bin"
INFO="$AMON_HOME/server_info.txt"
echo "[setup] AMON_HOME=$AMON_HOME"

# 1. Preflight: GPUs, driver, OS, disk, network. Everything goes to server_info.txt too.
select_gpus
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
DRIVER_MAJOR="${DRIVER%%.*}"
GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
DISK_GB="$(df -Pk "$AMON_HOME" | awk 'NR==2 {print int($4 / 1048576)}')"
REPO_DISK_GB="$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "os: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r), glibc $GLIBC"
  echo "cpus: $(nproc), memory: $(free -g | awk '/Mem:/ {print $2}') GB"
  echo "driver: $DRIVER"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  echo "free disk: $DISK_GB GB in $AMON_HOME, $REPO_DISK_GB GB in $PWD (runs/)"
  echo "compiler: $(command -v cc || command -v gcc || echo none)"
  echo "tmux: $(command -v tmux || echo none), screen: $(command -v screen || echo none)"
  echo "git commit: $(git rev-parse HEAD)"
} | tee "$INFO"

(( DRIVER_MAJOR >= MIN_DRIVER )) || fail "NVIDIA driver $DRIVER is older than $MIN_DRIVER (CUDA 12)"
[[ "$(printf "%s\n%s\n" "$MIN_GLIBC" "$GLIBC" | sort -V | head -1)" == "$MIN_GLIBC" ]] \
  || fail "glibc $GLIBC is older than $MIN_GLIBC (Ubuntu 20.04); the locked wheels need it"
(( DISK_GB >= WANT_DISK_GB )) \
  || echo "[setup] WARNING: only $DISK_GB GB free in $AMON_HOME; about $WANT_DISK_GB GB are needed"
(( REPO_DISK_GB >= 20 )) \
  || echo "[setup] WARNING: only $REPO_DISK_GB GB free for runs/; the pilot needs about 5 GB"
for url in https://pypi.org/simple/ https://huggingface.co https://github.com \
  https://astral.sh/uv/install.sh; do
  curl -sSfI --max-time 15 "$url" > /dev/null || fail "cannot reach $url (proxy or firewall?)"
done
echo "[setup] network ok"

# 2. uv, without sudo, into ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AMON_HOME/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$AMON_HOME/python}"
if ! command -v uv > /dev/null; then
  echo "[setup] installing uv $UV_VERSION into ~/.local/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
echo "[setup] $(uv --version)"

# 3. Simulation environment (reinstalled when its lock file or pyproject changes).
SIM_VENV="$AMON_HOME/venv"
WANTED="$(sha256sum requirements/server_3070.txt pyproject.toml)"
if [[ ! -x "$SIM_VENV/bin/python" || "$(cat "$AMON_HOME/sim.stamp" 2>/dev/null)" != "$WANTED" ]]; then
  echo "[setup] installing the simulation environment into $SIM_VENV"
  [[ -x "$SIM_VENV/bin/python" ]] || uv venv -q "$SIM_VENV" --python 3.11
  uv pip sync -p "$SIM_VENV/bin/python" requirements/server_3070.txt
  uv pip install -q -p "$SIM_VENV/bin/python" --no-deps -e .
  echo "$WANTED" > "$AMON_HOME/sim.stamp"
fi
echo "[setup] simulation environment ok"

# 4. VLM environment: the CUDA 13 build needs driver 580 or newer, else vLLM 0.11 on CUDA 12.8.
VLM_VENV="$AMON_HOME/venv-vlm"
if [[ -n "${VLM_LOCK:-}" ]]; then
  echo "[setup] VLM lock chosen by VLM_LOCK: $VLM_LOCK"
elif (( DRIVER_MAJOR >= CUDA13_DRIVER )); then
  VLM_LOCK="requirements/vlm_freeze.txt"
else
  VLM_LOCK="requirements/vlm_cu12.txt"
fi
CC_LINE=""
if [[ "${SKIP_VLM:-0}" != 1 ]]; then
  WANTED="$(sha256sum "$VLM_LOCK")"
  if [[ ! -x "$VLM_VENV/bin/python" || "$(cat "$AMON_HOME/vlm.stamp" 2>/dev/null)" != "$WANTED" ]]; then
    echo "[setup] installing the VLM environment from $VLM_LOCK into $VLM_VENV (about 10 GB)"
    [[ -x "$VLM_VENV/bin/python" ]] || uv venv -q "$VLM_VENV" --python 3.11
    uv pip sync -p "$VLM_VENV/bin/python" "$VLM_LOCK"
    echo "$WANTED" > "$AMON_HOME/vlm.stamp"
  fi
  # Triton compiles small launchers at server start and needs a C compiler; zig works without root.
  if ! command -v cc > /dev/null && ! command -v gcc > /dev/null; then
    uv pip install -q -p "$VLM_VENV/bin/python" ziglang==0.17.1
    printf '#!/usr/bin/env bash\nexec "%s" -m ziglang cc "$@"\n' "$VLM_VENV/bin/python" \
      > "$AMON_HOME/bin/zigcc"
    chmod +x "$AMON_HOME/bin/zigcc"
    CC_LINE="export CC=\"$AMON_HOME/bin/zigcc\""
    echo "[setup] no system C compiler; using zig ($AMON_HOME/bin/zigcc)"
  fi
  echo "[setup] VLM environment ok: $("$VLM_VENV/bin/python" -c 'import vllm; print("vllm", vllm.__version__)')"
  echo "vlm lock: $VLM_LOCK" >> "$INFO"
fi

# 5. env.sh, sourced by check.sh, run_3070_server.sh and interactive shells.
cat > "$AMON_HOME/env.sh" << EOF
# Written by server/setup.sh on $(date -I). Source it: source $AMON_HOME/env.sh
export AMON_HOME="$AMON_HOME"
export SIM_VENV="$SIM_VENV" VLM_VENV="$VLM_VENV"
export HF_HOME="${HF_HOME:-$AMON_HOME/hf}" TORCH_HOME="${TORCH_HOME:-$AMON_HOME/torch}"
export UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export MUJOCO_GL=egl CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="\$HOME/.local/bin:\$PATH"
$CC_LINE
# Extra vllm serve flags found necessary by server/check.sh (empty when none were).
if [[ -f "$AMON_HOME/vlm.env" ]]; then source "$AMON_HOME/vlm.env"; fi
EOF

# 6. Model weights.
load_env
SKIP_VLM_ARG=()
[[ "${SKIP_VLM:-0}" == 1 ]] && SKIP_VLM_ARG=(--skip-vlm)
python scripts/fetch_models.py --out "$AMON_HOME/model_revisions.json" "${SKIP_VLM_ARG[@]}"

echo
echo "[setup] done. Next: bash run_3070_server.sh --check"
