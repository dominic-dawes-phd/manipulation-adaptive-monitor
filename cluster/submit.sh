#!/usr/bin/env bash
# Submit a sweep as chained Slurm arrays, one per stage, plus a collect job.
# Usage: bash cluster/submit.sh configs/sweeps/main.yaml [NUM_SHARDS]
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

SPEC="$1"
NUM_SHARDS="${2:-8}"
NAME="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$SPEC")"
STAMP="${NAME}_$(date +%Y%m%d-%H%M%S)"
REPO_DIR="$(pwd)"
mkdir -p "$RUNS_DIR/$STAMP/logs"
cp "$SPEC" "$RUNS_DIR/$STAMP/spec.yaml"

COMMON=($SLURM_CLUSTER_FLAG --account="$ACCOUNT" --time="$TIME_PER_SHARD"
        --cpus-per-task="$CPUS_PER_SHARD" --mem="$MEM_PER_SHARD"
        --constraint="$GPU_CONSTRAINT" --parsable
        --output="$RUNS_DIR/$STAMP/logs/%x_%A_%a.out")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")

STAGES="$(python -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['stages']))" "$SPEC")"
DEPENDENCY=()
LAST_JOB=""
for STAGE in $STAGES; do
  ARRAY="0-$((NUM_SHARDS - 1))"
  [[ "$STAGE" == "train_wm" || "$STAGE" == "evaluate" || "$STAGE" == "plot" ]] && ARRAY="0"
  SHARDS_FOR_STAGE="$NUM_SHARDS"
  [[ "$ARRAY" == "0" ]] && SHARDS_FOR_STAGE=1
  LAST_JOB=$(sbatch "${COMMON[@]}" "${DEPENDENCY[@]}" --array="$ARRAY" --job-name="amon_${STAGE}" \
    --export=ALL,SPEC="$SPEC",STAGE="$STAGE",STAMP="$STAMP",NUM_SHARDS="$SHARDS_FOR_STAGE",REPO_DIR="$REPO_DIR" \
    cluster/job.sbatch)
  LAST_JOB="${LAST_JOB%%;*}"
  echo "[submit] $STAGE -> job $LAST_JOB (array $ARRAY)"
  DEPENDENCY=(--dependency=afterok:"$LAST_JOB")
done

sbatch $SLURM_CLUSTER_FLAG --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=2 --mem=8G \
  --dependency=afterany:"$LAST_JOB" --job-name=amon_collect \
  --output="$RUNS_DIR/$STAMP/logs/collect.out" \
  --wrap="bash $REPO_DIR/cluster/collect.sh $RUNS_DIR/$STAMP"
echo "[submit] run dir: $RUNS_DIR/$STAMP"
