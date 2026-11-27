#!/usr/bin/env bash
# Summarize a finished run: shard exit statuses, failed episodes, a rerun hint.
# Usage: bash cluster/collect.sh <run_dir>
set -euo pipefail
RUN_DIR="$1"
SUMMARY="$RUN_DIR/summary.txt"
{
  echo "run: $RUN_DIR"
  echo "collected: $(date -Is) on $(hostname)"
  echo
  echo "== shard statuses =="
  grep -h "status=" "$RUN_DIR"/logs/*.out 2>/dev/null || echo "(no status lines found)"
  echo
  echo "== failed episodes =="
  grep -hE "EPISODE_FAILED|SCORE_FAILED" "$RUN_DIR"/logs/*.out 2>/dev/null | sort -u || true
  FAILED_IDS=$(grep -h "EPISODE_FAILED" "$RUN_DIR"/logs/*.out 2>/dev/null \
    | sed -E 's/.*EPISODE_FAILED ([^ ]+).*/\1/' | sort -u | paste -sd, - || true)
  if [[ -n "$FAILED_IDS" ]]; then
    echo
    echo "rerun with spec key:  only: [$FAILED_IDS]"
  fi
  echo
  echo "== episode counts =="
  find "$RUN_DIR/episodes" -name meta.json 2>/dev/null | wc -l
} > "$SUMMARY"
cat "$SUMMARY"
if [[ -n "${MAIL_USER:-}" ]] && command -v mail >/dev/null; then
  mail -s "amon run done: $(basename "$RUN_DIR")" "$MAIL_USER" < "$SUMMARY"
fi
