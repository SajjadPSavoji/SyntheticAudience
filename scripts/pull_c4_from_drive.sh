#!/usr/bin/env bash
#
# Pull a C4 run from Google Drive down to this repo's data/results/<name> via rclone.
#
# The run was written on Colab under  <remote>:SyntheticAudience_C4/<run>/
# with subdirs  edits/ logs/ analysis/ .
#
# One-time setup:
#   brew install rclone      # (already present if this repo is set up)
#   rclone config            # create a remote of type "Google Drive"; name it e.g. gdrive
#
# Usage:
#   scripts/pull_c4_from_drive.sh [RUN] [DEST_NAME]
#     RUN        Drive run folder name          (default: c4_run1)
#     DEST_NAME  local folder under data/results (default: same as RUN)
#
# Env overrides:
#   RCLONE_REMOTE=gdrive                 # rclone remote name (default: gdrive)
#   DRIVE_BASE=SyntheticAudience_C4      # base folder on the remote (default: SyntheticAudience_C4)
#   SKIP_EDITS=1                         # pull logs/ + analysis/ only (skip the big PNG tree)
#   MIRROR=1                             # exact mirror (rclone sync, deletes extra local files);
#                                        # default is rclone copy (add/update only, never deletes)
#
# Examples:
#   scripts/pull_c4_from_drive.sh                       # c4_run1 -> data/results/c4_run1
#   scripts/pull_c4_from_drive.sh c4_run2               # c4_run2 -> data/results/c4_run2
#   SKIP_EDITS=1 scripts/pull_c4_from_drive.sh c4_run1  # skip edited images
set -euo pipefail

RUN="${1:-c4_run1}"
DEST_NAME="${2:-$RUN}"
REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_BASE="${DRIVE_BASE:-SyntheticAudience_C4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/results/$DEST_NAME"

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone not found. Install it:  brew install rclone" >&2
  exit 1
fi

# Verify the remote exists (rclone prints remotes as 'name:').
if ! rclone listremotes | grep -qx "${REMOTE}:"; then
  echo "ERROR: rclone remote '${REMOTE}:' is not configured." >&2
  echo "Configured remotes:" >&2
  rclone listremotes >&2 || true
  echo "Create one with:  rclone config   (type: Google Drive), then re-run with RCLONE_REMOTE=<name>." >&2
  exit 1
fi

SRC="${REMOTE}:${DRIVE_BASE}/${RUN}"

# Confirm the run exists on the remote; if not, list what's there.
if ! rclone lsf "${REMOTE}:${DRIVE_BASE}" 2>/dev/null | grep -qx "${RUN}/"; then
  echo "ERROR: run '${RUN}' not found at ${REMOTE}:${DRIVE_BASE}" >&2
  echo "Available runs:" >&2
  rclone lsf "${REMOTE}:${DRIVE_BASE}" >&2 || true
  exit 1
fi

CMD="copy"
[ "${MIRROR:-0}" = "1" ] && CMD="sync"
EXCLUDES=()
if [ "${SKIP_EDITS:-0}" = "1" ]; then
  EXCLUDES=(--exclude "edits/**")
  echo "SKIP_EDITS=1 -> pulling logs/ + analysis/ only"
fi

mkdir -p "$DEST"
echo "rclone ${CMD}:"
echo "  from: $SRC"
echo "  to:   $DEST"
rclone "$CMD" "$SRC" "$DEST" -P ${EXCLUDES[@]+"${EXCLUDES[@]}"}

echo
echo "Done. Re-analyze locally with:"
echo "  cd scripts/analysis"
echo "  python c4_trajectory.py  --output-root '$DEST'"
echo "  python c4_qualitative.py --output-root '$DEST'"
