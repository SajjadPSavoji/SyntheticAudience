#!/usr/bin/env bash
# Convenience wrapper for the claim-4 iterative persona-edit pipeline.
#
# Picks the two GPUs with the most free memory and pins them so that cuda:0 (the
# ~62 GB image editor) gets the roomier card and cuda:1 (the ~16 GB critic VLM)
# the other. Any extra args are forwarded to claim4_pipeline.py, e.g.:
#
#     bash script/claim4_pipeline.sh --n-images 10 --n-iterations 10
#     bash script/claim4_pipeline.sh --n-images 1 --n-iterations 2   # smoke test
#
# For a long run that must survive your VS Code / SSH session, launch under
# systemd-run --user, e.g.:
#     systemd-run --user --scope bash script/claim4_pipeline.sh --n-images 10
set -euo pipefail

cd "$(dirname "$0")/.."

# Respect a caller-set CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=7,5); only
# auto-pick the two most-free GPUs when it isn't already set.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -nr | awk -F',' '{gsub(/ /,"",$1); print $1}'
  )
  if [[ "${#GPUS[@]}" -ge 2 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}"
  else
    export CUDA_VISIBLE_DEVICES="${GPUS[0]:-0}"
  fi
fi
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (editor=cuda:0, critic=cuda:1)"

# Match the repo convention: run inside the persona conda env.
if [[ "${CONDA_DEFAULT_ENV:-}" != "persona" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate persona
fi

exec python script/claim4_pipeline.py "$@"
