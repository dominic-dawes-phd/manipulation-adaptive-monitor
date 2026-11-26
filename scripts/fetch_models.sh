#!/usr/bin/env bash
# Download every model this project uses into $HF_HOME and record their hashes.
# Usage: bash scripts/fetch_models.sh [HF_HOME]
set -euo pipefail
export HF_HOME="${1:-${HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_HOME"

models=(
  lerobot/act_aloha_sim_transfer_cube_human
  lerobot/act_aloha_sim_insertion_human
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ
  Qwen/Qwen2.5-VL-3B-Instruct-AWQ
  facebook/dinov2-small
)
# DINOv3 weights may require accepting a license on Hugging Face first.
# Add the exact repo id here after M0 (decision D-04).

for repo in "${models[@]}"; do
  echo "[fetch] $repo"
  huggingface-cli download "$repo" --quiet
done

echo "[fetch] recording commit hashes"
python - <<'EOF'
import json
from huggingface_hub import HfApi

repos = [
    "lerobot/act_aloha_sim_transfer_cube_human",
    "lerobot/act_aloha_sim_insertion_human",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "facebook/dinov2-small",
]
api = HfApi()
hashes = {repo: api.model_info(repo).sha for repo in repos}
with open("model_hashes.json", "w") as handle:
    json.dump(hashes, handle, indent=2)
print(json.dumps(hashes, indent=2))
EOF
