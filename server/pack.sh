#!/usr/bin/env bash
# Packs a finished run, episodes included, for copying to the laptop, with a checksum.
#
#   bash server/pack.sh runs/pilot_<stamp>              # writes runs/pilot_<stamp>.tar
#   bash server/pack.sh runs/main_<stamp> --no-frames   # without episodes/ (scores and models only)
#
# data.npz files are already compressed, so the archive is a plain tar.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1%/}"
[[ -f "$RUN_DIR/run.json" ]] || { echo "not a run folder: $RUN_DIR" >&2; exit 1; }
NAME="$(basename "$RUN_DIR")"
TARBALL="$(dirname "$RUN_DIR")/$NAME.tar"
EXCLUDES=()
[[ "${2:-}" == "--no-frames" ]] && EXCLUDES+=(--exclude="$NAME/episodes")
[[ -f "$RUN_DIR/summary.txt" ]] || echo "WARNING: no summary.txt; the run may not have finished"
tar -cf "$TARBALL" -C "$(dirname "$RUN_DIR")" "${EXCLUDES[@]}" "$NAME"
(cd "$(dirname "$TARBALL")" && sha256sum "$NAME.tar" > "$NAME.tar.sha256")
echo "packed: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo
echo "On the laptop (Git Bash or WSL, in the repository folder):"
echo "  scp $USER@$(hostname -f):$(realpath "$TARBALL")* runs/"
echo "  (cd runs && sha256sum -c $NAME.tar.sha256 && tar -xf $NAME.tar)"
