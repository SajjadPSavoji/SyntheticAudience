#!/usr/bin/env bash
#
# Work through the whole C4 unit list on a shared node, placing each unit on an
# allowed GPU as soon as that GPU has room. Wraps scripts/c4_shard.sh, so every
# unit still gets the same preflight, logging and --resume behaviour.
#
#   OUTPUT_ROOT=/shared/$USER/c4_runs/c4_run1 scripts/c4_autorun.sh
#
# Safe to stop (Ctrl-C) and restart: units already finished are skipped by
# --resume, and a unit killed mid-flight just resumes from its last checkpoint.
#
# Env overrides (with defaults):
#   OUTPUT_ROOT=/shared/$USER/c4_runs/c4_run1
#   GPUS=4,5,6,7           GPUs this run may use
#   SHARDS=4               image sub-shards per condition (4 conditions x SHARDS units)
#   CONDITIONS=society,blind,static,reward_only     (expensive ones first)
#   MAX_CONCURRENT=3       units running at once
#   POLL_SECS=120          how often to look for a free GPU
#   OFFLOAD_AFTER_SECS=1800  if a unit cannot be placed with resident weights for
#                          this long, allow CPU_OFFLOAD=1 placement (needs ~25 GB
#                          less VRAM but is much slower). 0 disables the fallback.
#   RUN_ANALYSIS=1         build figures + table once every unit is done
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-/shared/$USER/c4_runs/c4_run1}"
GPUS="${GPUS:-4,5,6,7}"
SHARDS="${SHARDS:-4}"
CONDITIONS="${CONDITIONS:-society,blind,static,reward_only}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
POLL_SECS="${POLL_SECS:-120}"
OFFLOAD_AFTER_SECS="${OFFLOAD_AFTER_SECS:-1800}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a COND_LIST <<< "$CONDITIONS"

# Build the unit queue: expensive conditions first so they get the big GPUs.
QUEUE=()
for cond in "${COND_LIST[@]}"; do
  for ((i = 0; i < SHARDS; i++)); do QUEUE+=("$cond $i/$SHARDS"); done
done

echo "=================== C4 autorun ==================="
echo "output_root    : $OUTPUT_ROOT"
echo "allowed GPUs   : ${GPU_IDS[*]}"
echo "units          : ${#QUEUE[@]}  (${#COND_LIST[@]} conditions x $SHARDS shards)"
echo "max concurrent : $MAX_CONCURRENT"
echo "offload after  : ${OFFLOAD_AFTER_SECS}s unplaced (0 = never)"
echo "=================================================="

mkdir -p "$OUTPUT_ROOT/stdout"
declare -A RUN_PID=()   # gpu -> pid of the unit occupying it
declare -A RUN_UNIT=()  # gpu -> unit description
started=$(date +%s)

reap() {  # drop finished workers so their GPU frees up
  local g
  for g in "${!RUN_PID[@]}"; do
    if ! kill -0 "${RUN_PID[$g]}" 2>/dev/null; then
      wait "${RUN_PID[$g]}" 2>/dev/null
      local rc=$?
      if [ "$rc" = "0" ]; then
        echo "[$(date +%T)] DONE  ${RUN_UNIT[$g]} (GPU $g)"
      else
        # Requeue: an OOM, a killed worker, or a preflight abort (the GPU filled
        # up between our probe and the launch) all just mean "try again later".
        # Read the reason from this unit's OWN log — every worker appends to the
        # shared placement.log, so its tail is usually some other unit's chatter.
        local why unit_log cond_ shard_ tag
        read -r cond_ shard_ <<< "${RUN_UNIT[$g]}"
        tag="$cond_"
        [ -n "$shard_" ] && tag="${cond_}_shard${shard_//\//of}"
        unit_log="$OUTPUT_ROOT/stdout/c4_${tag}_gpu${g}.log"
        why=$(grep -hE "ABORT|OutOfMemory|Error:" "$unit_log" \
                "$OUTPUT_ROOT/stdout/placement.log" 2>/dev/null | tail -1 | cut -c1-160)
        echo "[$(date +%T)] FAIL  ${RUN_UNIT[$g]} (GPU $g, exit $rc) — requeued${why:+: $why}"
        QUEUE+=("${RUN_UNIT[$g]}")
      fi
      unset 'RUN_PID[$g]' 'RUN_UNIT[$g]'
    fi
  done
}

n_running() { echo "${#RUN_PID[@]}"; }

unplaced_since=$(date +%s)

while [ "${#QUEUE[@]}" -gt 0 ] || [ "$(n_running)" -gt 0 ]; do
  reap

  # Allow the slow-but-small configuration only once we have been stuck a while.
  offload_ok=0
  if [ "$OFFLOAD_AFTER_SECS" != "0" ] && \
     [ $(( $(date +%s) - unplaced_since )) -ge "$OFFLOAD_AFTER_SECS" ]; then
    offload_ok=1
  fi

  placed_any=0
  while [ "${#QUEUE[@]}" -gt 0 ] && [ "$(n_running)" -lt "$MAX_CONCURRENT" ]; do
    # Scan the WHOLE queue, not just its head: society/blind need ~20 GB more
    # than static/reward_only, so a society unit that cannot be placed must not
    # block a static unit that fits on the GPU sitting idle right now.
    pick_idx=-1; pick_gpu=""; pick_offload=0
    for try_offload in 0 1; do
      [ "$try_offload" = "1" ] && [ "$offload_ok" != "1" ] && continue
      for idx in "${!QUEUE[@]}"; do
        read -r cond shard <<< "${QUEUE[$idx]}"
        for g in "${GPU_IDS[@]}"; do
          [ -n "${RUN_PID[$g]:-}" ] && continue
          # CONDA_ENV= skips the env activation: a DRY_RUN probe exits before it
          # would run python, and a full scan is up to 128 probes per cycle, so
          # activating conda each time costs ~60 s of pure overhead per cycle.
          if CONDA_ENV= OUTPUT_ROOT="$OUTPUT_ROOT" CPU_OFFLOAD="$try_offload" DRY_RUN=1 \
             scripts/c4_shard.sh "$g" "$cond" "$shard" >/dev/null 2>&1; then
            pick_idx="$idx"; pick_gpu="$g"; pick_offload="$try_offload"; break
          fi
        done
        [ -n "$pick_gpu" ] && break
      done
      [ -n "$pick_gpu" ] && break
    done
    [ -z "$pick_gpu" ] && break   # nothing fits right now; wait and retry

    unit="${QUEUE[$pick_idx]}"
    read -r cond shard <<< "$unit"
    QUEUE=("${QUEUE[@]:0:$pick_idx}" "${QUEUE[@]:$((pick_idx + 1))}")

    # Keep the launcher's own output: the VRAM preflight can still abort here
    # (memory moves between the probe above and this call), and that message is
    # written before c4_refine.py's log file exists, so /dev/null would lose it.
    OUTPUT_ROOT="$OUTPUT_ROOT" CPU_OFFLOAD="$pick_offload" \
      scripts/c4_shard.sh "$pick_gpu" "$cond" "$shard" \
      >> "$OUTPUT_ROOT/stdout/placement.log" 2>&1 &
    RUN_PID[$pick_gpu]=$!
    RUN_UNIT[$pick_gpu]="$unit"
    echo "[$(date +%T)] START $unit on GPU $pick_gpu (offload=$pick_offload, pid ${RUN_PID[$pick_gpu]})"
    placed_any=1
    unplaced_since=$(date +%s)
    sleep 90   # let it allocate before we size up the next GPU
  done

  if [ "${#QUEUE[@]}" -gt 0 ] && [ "$placed_any" = "0" ]; then
    echo "[$(date +%T)] waiting — ${#QUEUE[@]} unit(s) queued, $(n_running) running. $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')"
  fi
  sleep "$POLL_SECS"
done

echo "all units finished in $(( ($(date +%s) - started) / 60 )) min"

if [ "$RUN_ANALYSIS" = "1" ]; then
  echo "== deliverables =="
  ( cd scripts/analysis \
      && python c4_trajectory.py  --output-root "$OUTPUT_ROOT" \
      && python c4_qualitative.py --output-root "$OUTPUT_ROOT" )
  echo "  -> $OUTPUT_ROOT/analysis/{c4.json,c4_summary.md,figs/}"
fi
echo "Done."
