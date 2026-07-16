# Running the C4 experiment on an H100 node

This is a step-by-step runbook for the **C4 auto-refinement editing experiment** on a GPU node
(H100 / A100). No Colab, no notebook — plain scripts. Target run: **10 steps, 100 images,
4 conditions.**

**What C4 tests:** using a *society of personas* as the critic that drives a 10-step image-editing
loop produces better edits than (a) a single **blind VLM** critic, or (b) a fixed **"improve this
image"** string. Each step re-edits the *original* image from an accumulated instruction and keeps a
new candidate only if a held-out aesthetic model says it improved *and* it stays visually close to
the source (drift guardrail). Full rationale: `research_plan.md` §7, §8.4, §14.19.

---

## 0. Prerequisites (do these once, before touching the node)

1. **A Hugging Face account with two things:**
   - Access to the private dataset repos `savoji/EVA` and `savoji/PARA` (ask the owner to add you).
   - **Accepted the license for the gated model** `black-forest-labs/FLUX.1-Kontext-dev` — open its
     model page on huggingface.co and click *Agree*. Without this, the editor download 403s.
2. **An HF token** (https://huggingface.co/settings/tokens, read scope is enough).
3. **Node access** with at least one H100/A100 (80GB ideal; FLUX + Qwen-7B + CLIP/DINOv2 together
   need ~45GB, so one 80GB GPU per shard is comfortable).

---

## 1. Get the code

```bash
git clone <repo-url> SyntheticAudience
cd SyntheticAudience
```

## 2. Set up the environment

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx        # your token
FETCH_DATA=1 scripts/setup_c4.sh
```

`setup_c4.sh`:
- creates a venv (`.venv`, or uses `uv` if available),
- installs **torch** (default CUDA-12 wheel; works on H100) then `requirements-gpu.txt`,
- prints the GPUs it can see and verifies `diffusers` + the `editor` package import,
- checks HF auth,
- with `FETCH_DATA=1`, downloads EVA + PARA into `data/`.

**HPC variants:**
- Torch already provided by a module? `module load <cuda/torch>` then `SKIP_TORCH=1 scripts/setup_c4.sh`.
- Need a specific CUDA wheel? `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 scripts/setup_c4.sh`.
- Custom venv path: `VENV=/scratch/$USER/venvs/c4 scripts/setup_c4.sh`.

Then activate it:
```bash
source .venv/bin/activate
```

If you skipped `FETCH_DATA=1`, fetch data manually (needs `HF_TOKEN` + repo access):
```bash
python scripts/fetch_from_hf.py eva para
```

## 3. Smoke test (2 images, ~2 minutes)

Always do this first — it loads every model and writes a few edits, so failures surface fast.

```bash
python script/c4_refine.py --dataset eva --n-images 2 \
    --conditions static,blind,society --editor flux --steps 3 --candidates 2 \
    --output-root /scratch/$USER/c4_smoke
```

Expect: models load, `[static]/[blind]/[society] ... checkpoint` lines, `Done.`, and PNGs under
`/scratch/$USER/c4_smoke/edits/`. If this works, the full run will too.

## 4. Full run — 10 steps, 100 images

```bash
OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh
```

Defaults are exactly the target: **`STEPS=10`, `TOTAL_IMAGES=100`** (split 50/50 across EVA+PARA),
`CONDITIONS=static,blind,society,reward_only`, `CANDIDATES=3`, `EDITOR=flux`. The script:
- **auto-detects GPUs and shards the image set across all of them** (one shard per GPU, pinned via
  `CUDA_VISIBLE_DEVICES`), waits for all shards,
- is **`--resume`-safe**: re-run the exact same command to continue after a timeout/crash,
- then runs the deliverables (figures + summary table).

**Common overrides** (env vars):
```bash
NGPU=4            scripts/run_c4.sh   # force 4-way shard (else auto)
TOTAL_IMAGES=200  scripts/run_c4.sh   # bigger run
DATASET=eva       scripts/run_c4.sh   # one dataset only (then TOTAL_IMAGES is that dataset's count)
RUN_ANALYSIS=0    scripts/run_c4.sh   # skip figures (run them later, see §6)
CONDITIONS=static,blind,society scripts/run_c4.sh   # drop the reward_only oracle
```

**Long runs / SLURM:** wrap it so it survives disconnects, e.g.
```bash
nohup env OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh > c4_run1.out 2>&1 &
# or inside an sbatch script; the loop is resumable so requeue is safe.
```

## 5. Where the outputs go

Everything lands under `OUTPUT_ROOT`:

```
$OUTPUT_ROOT/
  edits/<condition>/<image_id>/step*_cand*.png, step*_best.png   # every candidate + committed best
  logs/c4_<condition>/*.json                                     # per-step records (sharded, resumable)
  analysis/
    c4.json            # all metrics (headline gains, AUC, win-rates, convergence, drift, diversity)
    c4_summary.md      # the main results table
    figs/
      c4_trajectory.png   # best-so-far aesthetic vs step, per condition (+CI)
      c4_headline.png     # mean gain per condition + gain-vs-drift scatter (reward-hack check)
      c4_drift.png        # identity retention vs step (guardrail)
      c4_diversity.png    # distinct complaints/step: society vs blind
      c4_qualitative.png  # source vs best-edit-per-condition grid
  stdout/                 # per-GPU run logs
```

The one-line headline is printed at the end and stored in `c4.json` under `society_vs_blind_auc`.

## 6. (Re)generate the deliverables anytime

```bash
cd scripts/analysis
python c4_trajectory.py  --output-root /scratch/$USER/c4_run1
python c4_qualitative.py --output-root /scratch/$USER/c4_run1
```

## 7. Getting results off the node

Just copy `$OUTPUT_ROOT/analysis` (small) for the figures + table; copy `edits/` too if you want the
images (large — thousands of PNGs). E.g. `rsync -a node:/scratch/$USER/c4_run1/analysis ./`.

---

## Compute expectation

100 images × 4 conditions × 10 steps × 3 candidates ≈ **12k FLUX edits** + the persona critiques.
On a single H100 that's roughly a few hours; sharding across *N* GPUs divides it by ~*N* (4 GPUs →
well under an hour). `society` and `reward_only` are the heaviest (society adds 10 persona critiques
per step).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `403` / gated repo when loading FLUX | Accept the `FLUX.1-Kontext-dev` license on HF and make sure `HF_TOKEN` is exported. |
| `HF_TOKEN is not set` during fetch | `export HF_TOKEN=...`; also confirm you have access to `savoji/EVA`, `savoji/PARA`. |
| CUDA OOM | Use one 80GB GPU per shard (don't over-subscribe); reduce `--candidates`; or add `EXTRA_ARGS=--cpu-offload` (slower). |
| torch can't see CUDA / wrong CUDA | Reinstall torch matching the node's CUDA: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX scripts/setup_c4.sh`, or `SKIP_TORCH=1` after `module load`. |
| Run died partway | Re-run the **same** `run_c4.sh` command — `--resume` skips finished images (logs + per-shard edit cache persist). |
| A shard failed but others ran | Inspect `$OUTPUT_ROOT/stdout/c4_gpu*.log`; re-running resumes only the missing work. |
| `No c4 logs found ...` from analysis | Point `--output-root` at the same dir you ran with; the loop must have produced `logs/c4_*` first. |
| Blank images in the qualitative grid | Run the analysis on the node where the edits live (paths are resolved from the local `edits/`). |

## What "success" looks like

`c4_summary.md` should show `society` with the highest mean final gain and a **win-rate over blind
and static > 50%** (ideally with the AUC CI excluding 0), while the `c4_drift.png` guardrail holds
(committed edits stay above the 0.85 identity cap) — i.e. society improves images rather than just
transforming them. Sanity-check a few panels in `c4_qualitative.png` by eye: the gains are real only
if the society column looks *better*, not merely *different*.
