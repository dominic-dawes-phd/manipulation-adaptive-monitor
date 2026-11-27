#!/usr/bin/env bash
# Runs the server pipeline on one UMBC chip node with 4 GPUs, as a batch job (no open terminal).
# Written for chip-gpu, partition gpu, account pi_cmat. Everything large lives in the group space,
# because the home quota is 500 MB.
#
# Before the first submit:
#   mkdir -p /umbc/rs/pi_cmat/users/ddawes2/amon/logs
# Submit from the repository root (the directory the job is submitted from is used):
#   sbatch server/chip_job.sbatch                      # setup, check, smoke, pilot
#   STEPS="setup check smoke" sbatch server/chip_job.sbatch
#   STEPS="pilot" sbatch server/chip_job.sbatch        # after smoke has passed
#
# Results from chip (L40S) are a different hardware group from the 8x RTX 3070 server and
# must not be pooled with it (D-09). The job refuses to run if its GPUs are of mixed models.
#SBATCH --job-name=amon
#SBATCH --clusters=chip-gpu
#SBATCH --partition=gpu
#SBATCH --account=pi_cmat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/umbc/rs/pi_cmat/users/ddawes2/amon/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Keep every cache off the 500 MB home quota.
export AMON_HOME=/umbc/rs/pi_cmat/users/ddawes2/amon
export XDG_CACHE_HOME="$AMON_HOME/cache"
# These GPUs are not RTX 3070s, so the scripts' 3070 check must be bypassed. The mixed-model check
# below keeps a run to one GPU model.
export ALLOW_OTHER_GPU=1

GPUS_IN_JOB="${CUDA_VISIBLE_DEVICES:?Slurm did not set CUDA_VISIBLE_DEVICES}"
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
if [[ "$(echo "$GPU_NAMES" | wc -l)" != 1 ]]; then
  echo "[chip] ERROR: the job's GPUs are of mixed models: $GPU_NAMES"
  exit 1
fi
echo "[chip] $(date -Is) on $(hostname): GPUs $GPUS_IN_JOB, $GPU_NAMES, AMON_HOME=$AMON_HOME"

STEPS="${STEPS:-setup check smoke pilot}"
for STEP in $STEPS; do
  echo "[chip] === $STEP $(date -Is) ==="
  case "$STEP" in
    setup)
      bash server/setup.sh
      ;;
    check)
      bash run_3070_server.sh --check "$GPUS_IN_JOB"
      ;;
    smoke | pilot)
      bash run_3070_server.sh "configs/sweeps/$STEP.yaml" "$GPUS_IN_JOB"
      bash server/pack.sh "$(ls -td "runs/${STEP}_"* | head -1)"
      ;;
    *)
      echo "[chip] ERROR: unknown step '$STEP' (use setup, check, smoke, pilot)"
      exit 1
      ;;
  esac
done
echo "[chip] $(date -Is) finished: $STEPS"
