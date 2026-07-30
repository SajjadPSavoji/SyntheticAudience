#!/usr/bin/env bash
#
# Run the C4 experiment as ONE WORKER PER CONDITION on a *shared* GPU node.
#
# scripts/run_c4.sh shards the image set across N GPUs, which needs every GPU to
# hold a full worker (~55 GB: FLUX + Qwen2-VL + CLIP + DINOv2). On a busy node
# that is often impossible. This variant instead gives each *condition* its own
# GPU and waits for capacity before launching, which helps because the two
# critic-free conditions (static, reward_only) never load Qwen2-VL and therefore
# need ~20 GB less than society/blind.
#
# Each worker is launched with --resume, so re-running this script is safe and
# picks up whatever is missing.
#
# Usage:
#   OUTPUT_ROOT=/shared/$USER/c4_runs/c4_run1 scripts/run_c4_conditions.sh
#
# Env overrides (with defaults):
#   OUTPUT_ROOT=<repo>/outputs/c4_run1   all outputs (edits/ logs/ analysis/) go here
#   GPUS=4,5,6,7        physical GPUs this run is allowed to use
#   STEPS=10            refinement steps
#   TOTAL_IMAGES=100    total source images (split across datasets)
#   DATASET=both        eva | para | both
#   CONDITIONS=static,blind,society,reward_only
#   CANDIDATES=3        edits generated per step
#   EDITOR=flux         flux | instructpix2pix
#   CPU_OFFLOAD=1       stream FLUX weights (much smaller VRAM peak, ~20% slower)
#   NEED_VLM_MIB        free VRAM required to start society/blind
#   NEED_PLAIN_MIB      free VRAM required to start static/reward_only
#   WAIT_SECS=120       poll interval while waiting for a GPU
#   MAX_WAIT_SECS=86400 give up waiting for a condition after this long
#   RUN_ANALYSIS=1      run c4_trajectory.py + c4_qualitative.py at the end
#   EXTRA_ARGS=""       extra flags forwarded to script/c4_refine.py
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root

OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/outputs/c4_run1}"
GPUS="${GPUS:-4,5,6,7}"
STEPS="${STEPS:-10}"
TOTAL_IMAGES="${TOTAL_IMAGES:-100}"
DATASET="${DATASET:-both}"
CONDITIONS="${CONDITIONS:-static,blind,society,reward_only}"
CANDIDATES="${CANDIDATES:-3}"
EDITOR="${EDITOR:-flux}"
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
WAIT_SECS="${WAIT_SECS:-120}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-86400}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Free-VRAM needed before we start a worker. With CPU_OFFLOAD=1 the FLUX peak is
# the transformer alone (~24 GB) instead of the whole pipeline (~34 GB); the
# society/blind figures add the resident Qwen2-VL-7B (~18 GB).
if [ "$CPU_OFFLOAD" = "1" ]; then
  NEED_VLM_MIB="${NEED_VLM_MIB:-52000}"
  NEED_PLAIN_MIB="${NEED_PLAIN_MIB:-32000}"
else
  NEED_VLM_MIB="${NEED_VLM_MIB:-64000}"
  NEED_PLAIN_MIB="${NEED_PLAIN_MIB:-44000}"
fi

if [ "$DATASET" = "both" ]; then
  PER=$(( (TOTAL_IMAGES + 1) / 2 ))
else
  PER="$TOTAL_IMAGES"
fi

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a COND_LIST <<< "$CONDITIONS"

STDOUT_DIR="$OUTPUT_ROOT/stdout"
mkdir -p "$STDOUT_DIR"

echo "============== C4 run (one worker per condition) =============="
echo "output_root : $OUTPUT_ROOT"
echo "steps       : $STEPS"
echo "images      : $TOTAL_IMAGES total  ($PER per dataset x $DATASET)"
echo "conditions  : $CONDITIONS"
echo "candidates  : $CANDIDATES   editor: $EDITOR   cpu_offload: $CPU_OFFLOAD"
echo "allowed GPUs: ${GPU_IDS[*]}"
echo "need free   : ${NEED_VLM_MIB} MiB (society/blind), ${NEED_PLAIN_MIB} MiB (static/reward_only)"
echo "==============================================================="

COMMON=(--dataset "$DATASET" --n-images "$PER" --editor "$EDITOR"
        --steps "$STEPS" --candidates "$CANDIDATES"
        --output-root "$OUTPUT_ROOT" --resume)
[ "$CPU_OFFLOAD" = "1" ] && COMMON+=(--cpu-offload)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && COMMON+=($EXTRA_ARGS)

# GPUs currently occupied by a worker this script started (they have not shown up
# in nvidia-smi's free-memory figure yet while the model is still loading).
declare -A CLAIMED=()

free_mib() {  # $1 = gpu id
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

# Echo the first allowed GPU with >= $1 MiB free that we have not just claimed.
pick_gpu() {
  local need="$1" g f
  for g in "${GPU_IDS[@]}"; do
    [ -n "${CLAIMED[$g]:-}" ] && continue
    f=$(free_mib "$g")
    [ -z "$f" ] && continue
    if [ "$f" -ge "$need" ]; then echo "$g"; return 0; fi
  done
  return 1
}

start=$(date +%s)
pids=(); pid_cond=(); pid_gpu=()

for cond in "${COND_LIST[@]}"; do
  case "$cond" in
    society|blind) need="$NEED_VLM_MIB" ;;
    *)             need="$NEED_PLAIN_MIB" ;;
  esac

  # Reap finished workers so their GPU becomes claimable again.
  waited=0
  gpu=""
  while : ; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        unset 'CLAIMED[${pid_gpu[$i]}]'
      fi
    done
    if gpu=$(pick_gpu "$need"); then break; fi
    if [ "$waited" -ge "$MAX_WAIT_SECS" ]; then
      echo "SKIP [$cond]: no allowed GPU reached ${need} MiB free within ${MAX_WAIT_SECS}s" >&2
      break
    fi
    echo "  [$cond] waiting for a GPU with >= ${need} MiB free ($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '))"
    sleep "$WAIT_SECS"
    waited=$((waited + WAIT_SECS))
  done
  [ -z "$gpu" ] && continue

  CLAIMED[$gpu]=1
  log="$STDOUT_DIR/c4_${cond}_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python script/c4_refine.py "${COMMON[@]}" \
      --conditions "$cond" > "$log" 2>&1 &
  pid=$!
  pids+=("$pid"); pid_cond+=("$cond"); pid_gpu+=("$gpu")
  echo "  launched [$cond] on GPU $gpu (pid $pid)  log: $log"
  # Let the worker allocate before we measure free memory for the next one.
  sleep 90
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  [${pid_cond[$i]}] on GPU ${pid_gpu[$i]}: OK"
  else
    echo "  [${pid_cond[$i]}] on GPU ${pid_gpu[$i]}: FAILED (see $STDOUT_DIR/c4_${pid_cond[$i]}_gpu${pid_gpu[$i]}.log)" >&2
    fail=1
  fi
done
echo "run finished in $(( ($(date +%s) - start) / 60 )) min"

if [ "$RUN_ANALYSIS" = "1" ]; then
  echo "== deliverables (figures + summary table) =="
  ( cd scripts/analysis \
      && python c4_trajectory.py  --output-root "$OUTPUT_ROOT" \
      && python c4_qualitative.py --output-root "$OUTPUT_ROOT" )
  echo "  -> $OUTPUT_ROOT/analysis/{c4.json,c4_summary.md,figs/}"
fi

[ "$fail" = "1" ] && echo "NOTE: at least one condition failed — re-run this script to resume." >&2
echo "Done."
exit "$fail"
