#!/usr/bin/env bash
#
# Show C4 run progress: how many images each (condition, shard) has finished,
# which units are still running, and what is left to launch.
#
#   scripts/c4_status.sh [OUTPUT_ROOT]
#
# Env: OUTPUT_ROOT=/shared/$USER/c4_runs/c4_run1  (or pass it as $1)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-/shared/$USER/c4_runs/c4_run1}}"
LOGS="$OUTPUT_ROOT/logs"

echo "output_root : $OUTPUT_ROOT"
if [ ! -d "$LOGS" ]; then
  echo "no logs yet — nothing has been run into this root."
  exit 0
fi

echo
echo "images finished per condition/shard  (target: 100 images per condition)"
echo "-----------------------------------------------------------------------"
python - "$LOGS" <<'PY'
import glob, json, os, sys, collections
logs = sys.argv[1]
total = collections.Counter()
for cond in ("static", "blind", "society", "reward_only"):
    run = f"c4_{cond}"
    summaries = [s for s in glob.glob(os.path.join(logs, run, f"{run}*.json"))
                 if ".part-" not in os.path.basename(s)]
    if not summaries:
        print(f"  {cond:12s}  (not started)")
        continue
    for s in sorted(summaries):
        ids = set()
        base = s[:-len(".json")]
        for p in sorted(glob.glob(base + ".part-*.json")):
            try:
                for row in json.load(open(p, encoding="utf-8")):
                    ids.add(row["image_id"])
            except Exception as e:
                print(f"  !! unreadable {os.path.basename(p)}: {e}")
        tag = os.path.basename(s)[len(run):-len(".json")] or " (all images)"
        print(f"  {cond:12s}{tag:16s} {len(ids):4d} images")
        total[cond] += len(ids)
    print(f"  {cond:12s}{'TOTAL':16s} {total[cond]:4d} images")
PY

echo
echo "running now"
echo "-----------"
# pgrep + /proc, not `ps -o cmd`: on a node with thousands of processes the
# latter takes minutes. Shell wrappers repeat the worker's command line inside
# their -c string, so keep only processes whose argv[0] is the interpreter.
found=0
for pid in $(pgrep -u "$USER" -f "c4_refine\.py" 2>/dev/null); do
  mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || continue
  [ "${#argv[@]}" -gt 0 ] || continue
  case "${argv[0]}" in *python*) ;; *) continue ;; esac
  cmd="${argv[*]}"
  cond=$(sed -E 's/.*--conditions ([a-z_]+).*/\1/' <<<"$cmd")
  shard=$(sed -E 's/.*--shard ([0-9]+\/[0-9]+).*/\1/;t;s/.*/all/' <<<"$cmd")
  gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  echo "  pid $pid  up ${etime:-?}  condition=$cond shard=$shard gpu=${gpu:-?}"
  found=1
done
[ "$found" = "0" ] && echo "  (none)"

echo
echo "GPUs"
echo "----"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader \
  | awk -F', ' '{printf "  GPU%s  free=%-11s util=%s\n",$1,$2,$3}'

echo
echo "edited PNGs written: $(find "$OUTPUT_ROOT/edits" -name '*.png' 2>/dev/null | wc -l)"
echo "disk used          : $(du -sh "$OUTPUT_ROOT" 2>/dev/null | cut -f1)"
