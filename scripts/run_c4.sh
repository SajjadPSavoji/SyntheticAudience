#!/usr/bin/env bash
#
# Run the full C4 auto-refinement experiment on a GPU node (H100/A100).
# Defaults: 10 steps, 100 images total, 4 conditions, FLUX.1-Kontext editor.
# Auto-detects GPUs and shards the image set across them (one shard per GPU).
#
# Usage:
#   scripts/run_c4.sh                                  # 10 steps, 100 images, all GPUs
#   OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh
#   NGPU=4 scripts/run_c4.sh                           # force 4-way shard
#   TOTAL_IMAGES=200 STEPS=10 scripts/run_c4.sh
#   RUN_ANALYSIS=0 scripts/run_c4.sh                   # skip the figures/table step
#
# Env overrides (with defaults):
#   OUTPUT_ROOT=<repo>/outputs/c4_run1   all outputs (edits/ logs/ analysis/) go here
#   STEPS=10            refinement steps
#   TOTAL_IMAGES=100    total source images (split across datasets)
#   DATASET=both        eva | para | both
#   CONDITIONS=static,blind,society,reward_only
#   CANDIDATES=3        edits generated per step
#   EDITOR=flux         flux | instructpix2pix
#   NGPU=auto           number of GPUs to shard across (auto = detect)
#   RUN_ANALYSIS=1      run c4_trajectory.py + c4_qualitative.py after the run
#   EXTRA_ARGS=""       extra flags forwarded to script/c4_refine.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root

OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/outputs/c4_run1}"
STEPS="${STEPS:-10}"
TOTAL_IMAGES="${TOTAL_IMAGES:-100}"
DATASET="${DATASET:-both}"
CONDITIONS="${CONDITIONS:-static,blind,society,reward_only}"
CANDIDATES="${CANDIDATES:-3}"
EDITOR="${EDITOR:-flux}"
NGPU="${NGPU:-auto}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Split TOTAL_IMAGES across datasets so the flag (which is per-dataset) honors the total.
if [ "$DATASET" = "both" ]; then
  PER=$(( (TOTAL_IMAGES + 1) / 2 ))
else
  PER="$TOTAL_IMAGES"
fi

# Detect GPU count.
if [ "$NGPU" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NGPU=1
  fi
fi
[ "${NGPU:-0}" -lt 1 ] && NGPU=1

STDOUT_DIR="$OUTPUT_ROOT/stdout"
mkdir -p "$STDOUT_DIR"

echo "==================== C4 run ===================="
echo "output_root : $OUTPUT_ROOT"
echo "steps       : $STEPS"
echo "images      : $TOTAL_IMAGES total  ($PER per dataset x $DATASET)"
echo "conditions  : $CONDITIONS"
echo "candidates  : $CANDIDATES   editor: $EDITOR"
echo "GPUs        : $NGPU"
echo "================================================"

# Common args to script/c4_refine.py. --resume makes the whole thing idempotent.
COMMON=(--dataset "$DATASET" --n-images "$PER" --conditions "$CONDITIONS"
        --editor "$EDITOR" --steps "$STEPS" --candidates "$CANDIDATES"
        --output-root "$OUTPUT_ROOT" --resume)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && COMMON+=($EXTRA_ARGS)

start=$(date +%s)
if [ "$NGPU" -le 1 ]; then
  echo "single-GPU run -> $STDOUT_DIR/c4.log"
  python script/c4_refine.py "${COMMON[@]}" 2>&1 | tee "$STDOUT_DIR/c4.log"
else
  echo "sharding across $NGPU GPUs (image round-robin, one shard per GPU)"
  pids=()
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES="$g" python script/c4_refine.py "${COMMON[@]}" \
        --shard "$g/$NGPU" > "$STDOUT_DIR/c4_gpu$g.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "  shard $g/$NGPU -> GPU $g (pid $pid)  log: $STDOUT_DIR/c4_gpu$g.log"
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [ "$fail" = "1" ]; then
    echo "ERROR: at least one shard failed; inspect $STDOUT_DIR/c4_gpu*.log" >&2
    exit 1
  fi
fi
echo "run finished in $(( ($(date +%s) - start) / 60 )) min"

if [ "$RUN_ANALYSIS" = "1" ]; then
  echo "== deliverables (figures + summary table) =="
  ( cd scripts/analysis \
      && python c4_trajectory.py  --output-root "$OUTPUT_ROOT" \
      && python c4_qualitative.py --output-root "$OUTPUT_ROOT" )
  echo "  -> $OUTPUT_ROOT/analysis/{c4.json,c4_summary.md,figs/}"
fi
echo "Done."
