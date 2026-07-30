#!/usr/bin/env bash
#
# Run ONE C4 unit of work — a single (condition, image-shard) pair — on ONE GPU
# you pick by hand. Built for a busy shared node: you launch a unit whenever a
# GPU frees up, and every unit is independent and --resume-safe.
#
#   scripts/c4_shard.sh <GPU> <CONDITION> [i/N]
#
#     GPU        physical GPU id, e.g. 6
#     CONDITION  static | blind | society | reward_only
#     i/N        optional image sub-shard, 0-indexed (e.g. 0/4). Omit = all images.
#
# Examples:
#   scripts/c4_shard.sh 6 society 0/4     # quarter of the images, society critic
#   scripts/c4_shard.sh 3 static          # all images, static critic
#
# Re-running the same command resumes: finished images are skipped, and the
# per-shard edit cache means already-generated FLUX candidates are not redone.
# Units never collide — each writes its own log file and its own edit cache.
#
# Env overrides (with defaults):
#   OUTPUT_ROOT=/shared/$USER/c4_runs/c4_run1   where edits/ logs/ analysis/ go
#   CONDA_ENV=persona        conda env to activate (set to "" if already active)
#   C4_HF_HOME=/shared/amin/hf_cache            model cache (has FLUX/Qwen/CLIP/DINOv2).
#                            NOTE: an inherited HF_HOME is ignored on purpose — see below.
#   STEPS=10  CANDIDATES=3  DATASET=both  PER_DATASET=50   the target defaults
#   GUIDANCE=2.5             editor guidance scale; higher = bolder edits
#   EMPHASIS=""              prompt suffix forcing a visible edit; "" disables it
#                            (see the note below on why both are toned down)
#   CPU_OFFLOAD=0            1 = stream FLUX weights; fits ~25 GB less but ~7x slower
#   CHECKPOINT_INTERVAL=1    flush results every N finished images (see note below)
#   MIN_FREE_MIB=            override the free-VRAM preflight threshold
#   FORCE=0                  1 = skip the free-VRAM preflight entirely
#   DRY_RUN=0                1 = print the command that would run, then exit
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root

if [ "$#" -lt 2 ]; then
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

GPU="$1"
COND="$2"
SHARD="${3:-}"

case "$COND" in
  static|blind|society|reward_only) ;;
  *) echo "ERROR: unknown condition '$COND' (static|blind|society|reward_only)" >&2; exit 2 ;;
esac
if [ -n "$SHARD" ] && ! [[ "$SHARD" =~ ^[0-9]+/[0-9]+$ ]]; then
  echo "ERROR: shard must look like i/N (e.g. 0/4), got '$SHARD'" >&2; exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/shared/$USER/c4_runs/c4_run1}"
CONDA_ENV="${CONDA_ENV-persona}"
# Deliberately NOT ${HF_HOME:-...}: this account's profile already exports
# HF_HOME=/shared/huggingface, a cache we cannot write to (CLIP is owned by
# another user, the FLUX lock dir by a third), which makes model loading die
# with PermissionError. Always point at our own cache; override with C4_HF_HOME.
export HF_HOME="${C4_HF_HOME:-/shared/amin/hf_cache}"
STEPS="${STEPS:-10}"
CANDIDATES="${CANDIDATES:-3}"
DATASET="${DATASET:-both}"
PER_DATASET="${PER_DATASET:-50}"     # 50 EVA + 50 PARA = the 100-image target
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Flush finished images to disk every N, rather than c4_refine.py's own default
# of 10. On this shared node a unit is regularly killed mid-flight by a
# neighbour's OOM, and with N=10 a resumed unit must finish ten more images
# before anything becomes durable -- society 1/4 spent 13 h restarting eight
# times and never once reached that boundary, so its checkpoint never moved off
# the 10 images it had banked at 04:22. N=1 makes every image durable, capping
# the loss from a kill at one image. It only changes write frequency: --resume
# skips completed images identically, and at <=275 rows per shard the results
# still fit in part-0001 whatever the interval.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"

# ~9 GB of our 77.6 GB peak was reserved-but-unallocated fragmentation when the
# last OOM hit. Expandable segments hand that back instead of stranding it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Edit boldness — the documented defaults from RUNNING_ON_H100.md.
#
# These briefly looked broken: the first run committed 1 candidate in 45, with
# drift as low as 0.11 and "improving" edits rejected by the 0.78 cap. That was
# the square-output bug in src/editor/flux_editor.py, not the parameters — a
# native-aspect source compared against a 1024x1024 edit both lost up to 0.47
# aesthetic points and inflated apparent DINOv2 drift. With that fixed, an A/B on
# identical images put guidance 3.0 + emphasis (gain +0.382, drift 0.945-0.979)
# level with guidance 2.5 + no emphasis (+0.376), so the documented defaults
# stand and the bug fix is the only deviation from the pre-registered setup.
# GUIDANCE=2.5 EMPHASIS="" reproduces the gentler variant.
GUIDANCE="${GUIDANCE:-3.0}"
EMPHASIS="${EMPHASIS-Make this a clearly visible edit, not a subtle one, while keeping the same subject and scene.}"

# How much free VRAM this unit needs before it is worth starting. society/blind
# keep Qwen2-VL-7B resident (~18 GB) on top of FLUX; static/reward_only never
# load it. --cpu-offload trades ~25 GB of VRAM for roughly a 7x slowdown.
if [ "$CPU_OFFLOAD" = "1" ]; then
  case "$COND" in society|blind) DEFAULT_NEED=52000 ;; *) DEFAULT_NEED=32000 ;; esac
else
  case "$COND" in society|blind) DEFAULT_NEED=62000 ;; *) DEFAULT_NEED=44000 ;; esac
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-$DEFAULT_NEED}"

if [ -n "$CONDA_ENV" ]; then
  # shellcheck disable=SC1091
  source /shared/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate conda env '$CONDA_ENV'" >&2; exit 1; }
fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
if [ -z "$FREE" ]; then
  echo "ERROR: GPU $GPU not visible to nvidia-smi" >&2; exit 1
fi
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
echo "GPU $GPU: ${FREE} MiB free, ${UTIL}% util   (this unit wants >= ${MIN_FREE_MIB} MiB)"

# Neighbours on this node grow their allocation over time (vLLM / RL rollout
# workers especially), and loading FLUX takes ~90 s. A single free-memory
# reading is therefore not enough: sample again and use the worse of the two,
# so a GPU that is actively filling up is rejected before we waste the load.
if [ "$FREE" -ge "$MIN_FREE_MIB" ] && [ "$FORCE" != "1" ] && [ "$DRY_RUN" != "1" ]; then
  sleep 10
  FREE2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  if [ -n "$FREE2" ] && [ "$FREE2" -lt "$FREE" ]; then
    echo "  re-check after 10s: ${FREE2} MiB free (dropped $((FREE - FREE2)) MiB — a neighbour is growing)"
    FREE="$FREE2"
  fi
fi

if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
  if [ "$FORCE" = "1" ]; then
    echo "WARNING: below threshold but FORCE=1 — starting anyway (may OOM)."
  else
    echo "ABORT: not enough free VRAM on GPU $GPU. Wait, or pick another GPU." >&2
    if [ "$CPU_OFFLOAD" != "1" ]; then
      echo "       CPU_OFFLOAD=1 needs ~25 GB less (but runs ~7x slower)." >&2
    fi
    echo "       MIN_FREE_MIB=<n> lowers this threshold; FORCE=1 skips the check entirely." >&2
    exit 1
  fi
fi
if [ "${UTIL:-0}" -ge 80 ]; then
  echo "NOTE: GPU $GPU is at ${UTIL}% utilization from another job — expect to run slower."
fi

TAG="$COND"
ARGS=(--dataset "$DATASET" --n-images "$PER_DATASET" --conditions "$COND"
      --editor flux --steps "$STEPS" --candidates "$CANDIDATES"
      --guidance-scale "$GUIDANCE" --edit-emphasis "$EMPHASIS"
      --checkpoint-interval "$CHECKPOINT_INTERVAL"
      --output-root "$OUTPUT_ROOT" --resume)
if [ -n "$SHARD" ]; then
  ARGS+=(--shard "$SHARD")
  # "0/4" -> "0of4", matching the shard tag c4_refine.py puts on its own logs.
  TAG="${COND}_shard${SHARD//\//of}"
fi
[ "$CPU_OFFLOAD" = "1" ] && ARGS+=(--cpu-offload)

mkdir -p "$OUTPUT_ROOT/stdout"
LOG="$OUTPUT_ROOT/stdout/c4_${TAG}_gpu${GPU}.log"

echo "condition   : $COND${SHARD:+   shard $SHARD}"
echo "output_root : $OUTPUT_ROOT"
echo "steps/cands : $STEPS / $CANDIDATES   images: $PER_DATASET per dataset x $DATASET"
echo "guidance    : $GUIDANCE   emphasis: ${EMPHASIS:-(disabled)}"
echo "cpu_offload : $CPU_OFFLOAD   checkpoint every $CHECKPOINT_INTERVAL image(s)"
echo "log         : $LOG"
echo "--------------------------------------------------------------"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1 — would execute:"
  echo "  HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=$GPU \\"
  echo "    python script/c4_refine.py ${ARGS[*]}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" python script/c4_refine.py "${ARGS[@]}" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
if [ "$rc" = "0" ]; then
  echo "OK: [$COND${SHARD:+ $SHARD}] finished on GPU $GPU."
else
  echo "FAILED (exit $rc): re-run the exact same command to resume from where it stopped." >&2
fi
exit "$rc"
