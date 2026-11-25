#!/usr/bin/env bash
# Pack a run's results (without raw frames) for copying to a laptop.
# Usage: bash cluster/pack_results.sh <run_dir> [--with-episodes]
set -euo pipefail
RUN_DIR="${1%/}"
NAME="$(basename "$RUN_DIR")"
TARBALL="$(dirname "$RUN_DIR")/${NAME}_results.tar.gz"
EXCLUDES=(--exclude="$NAME/episodes")
[[ "${2:-}" == "--with-episodes" ]] && EXCLUDES=()
tar -czf "$TARBALL" -C "$(dirname "$RUN_DIR")" "${EXCLUDES[@]}" "$NAME"
echo "packed: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "copy:   scp $USER@$(hostname -f):$(realpath "$TARBALL") ."
