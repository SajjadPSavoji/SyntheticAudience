#!/usr/bin/env bash
#
# One-command environment setup for the C4 experiments on a GPU node (H100/A100).
# Creates a venv, installs torch + the GPU stack, verifies CUDA, checks HF auth,
# and (optionally) fetches the EVA/PARA data.
#
# Usage:
#   scripts/setup_c4.sh                 # full setup
#   FETCH_DATA=1 scripts/setup_c4.sh    # also download EVA + PARA
#   SKIP_TORCH=1 scripts/setup_c4.sh    # skip torch (use a preloaded module/env)
#
# Env overrides:
#   VENV=.venv           # venv location
#   PYTHON=python3       # base interpreter
#   TORCH_INDEX_URL=...  # e.g. https://download.pytorch.org/whl/cu124  (match your CUDA)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root

VENV="${VENV:-.venv}"
PYTHON="${PYTHON:-python3}"
FETCH_DATA="${FETCH_DATA:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"

echo "== C4 environment setup =="
echo "repo : $(pwd)"
echo "venv : $VENV"

# 1) create + activate venv (prefer uv if available; it's much faster)
if command -v uv >/dev/null 2>&1; then
  echo "[1/5] creating venv with uv"
  uv venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PIP=(uv pip install)
else
  echo "[1/5] creating venv with $PYTHON -m venv"
  "$PYTHON" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  PIP=(pip install)
fi

# 2) torch (separate so it matches your CUDA)
if [ "$SKIP_TORCH" = "1" ]; then
  echo "[2/5] SKIP_TORCH=1 -> assuming torch is already available"
else
  echo "[2/5] installing torch"
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "${PIP[@]}" torch --index-url "$TORCH_INDEX_URL"
  else
    "${PIP[@]}" torch    # default PyPI wheel (CUDA 12.x; works on H100/A100)
  fi
fi

# 3) the rest of the GPU stack
echo "[3/5] installing requirements-gpu.txt"
"${PIP[@]}" -r requirements-gpu.txt

# 4) verify torch + CUDA + diffusers
echo "[4/5] verifying GPU stack"
python - <<'PY'
import torch, diffusers, transformers
print("  torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU{i}: {p.name}  {p.total_memory/1e9:.0f} GB")
else:
    print("  WARNING: no CUDA visible — FLUX will not run.")
print("  diffusers", diffusers.__version__, "| transformers", transformers.__version__)
import sys; sys.path.insert(0, "src")
import editor  # deferred-safe import of the C4 package (no torch pulled)
print("  editor package import OK")
PY

# 5) HF auth (FLUX.1-Kontext-dev is gated; data fetch also needs a token)
echo "[5/5] Hugging Face auth check"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "  HF_TOKEN is set."
elif python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "  huggingface-cli login detected."
else
  echo "  WARNING: no HF auth found."
  echo "  Run 'huggingface-cli login' or 'export HF_TOKEN=hf_...' before running."
  echo "  Required for the gated FLUX.1-Kontext-dev model and EVA/PARA download."
fi

if [ "$FETCH_DATA" = "1" ]; then
  echo "== fetching EVA + PARA =="
  python scripts/fetch_from_hf.py eva para
fi

echo
echo "Setup complete. Next:"
echo "  source $VENV/bin/activate"
echo "  scripts/run_c4.sh          # 10 steps, 100 images (see script header for knobs)"
