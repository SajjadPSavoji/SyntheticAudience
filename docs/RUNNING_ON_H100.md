# Running the C4 experiment on an H100 node

A complete, copy-paste runbook: **(A) set up the repo on the node → (B) what the experiment is →
(C) exactly what to run → (D) outputs → (E) zip and send.** No Colab, no notebook — plain scripts.

The result of a run is **logs (JSON) + edited images (PNG) + analysis figures**, all under one
`OUTPUT_ROOT`. At the end you zip that folder and send it over manually (Part E).

---

# Part A — Set up the repo on the node

## A.0 Prerequisites (once, per person)

1. **Hugging Face account with:**
   - read access to the private dataset repos **`savoji/PARA`** and **`savoji/EVA`** (ask the owner to add you), and
   - **the gated model license accepted** for **`black-forest-labs/FLUX.1-Kontext-dev`** — open its model page on huggingface.co and click *Agree* (otherwise the editor download 403s).
2. **An HF token** (https://huggingface.co/settings/tokens — read scope is enough).
3. **A GPU node** with ≥1 H100/A100. 80 GB is ideal: FLUX + Qwen2-VL-7B + CLIP + DINOv2 together need ~45 GB, so **one 80 GB GPU per shard** is comfortable.

## A.1 Clone

```bash
git clone <repo-url> SyntheticAudience
cd SyntheticAudience
```

## A.2 Environment + data (one command)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
FETCH_DATA=1 scripts/setup_c4.sh
source .venv/bin/activate
```

`setup_c4.sh` creates `.venv`, installs torch + `requirements-gpu.txt`, prints the visible GPUs,
verifies `diffusers`/`editor` import, checks HF auth, and (with `FETCH_DATA=1`) downloads PARA + EVA
into `data/`.

**HPC variants:**
- torch from a module: `module load <cuda/torch>` then `SKIP_TORCH=1 scripts/setup_c4.sh`
- specific CUDA wheel: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 scripts/setup_c4.sh`
- venv elsewhere: `VENV=/scratch/$USER/venvs/c4 scripts/setup_c4.sh`
- fetch data separately: `python scripts/fetch_from_hf.py para eva`

---

# Part B — What the experiment is

**Claim 4:** a *society of personas* used as the editing **critic** gives a better feedback signal
than a single **blind VLM** critic or a fixed **"improve this image"** string. We test it as a
**critic-quality ablation inside a 10-step auto-refinement editing loop** (rationale:
`research_plan.md` §7, §8.4, §14.19).

**The loop** (per image, 10 steps): the critic looks at the current best image → its complaints are
distilled into one edit instruction → **FLUX.1-Kontext** re-edits the *original* image with the
accumulated instruction (anchored, K candidates) → each candidate is scored by a **held-out LAION
aesthetic model** (a different model family from the Qwen critic — *critic ≠ objective*) and a
**DINOv2 identity/drift** check → the loop **commits a candidate only if it improves and stays above
the drift cap** (accept-if-better). Best-so-far is monotone by construction.

**The four conditions** (the ablation): `static` (fixed string) · `blind` (one generic VLM critique)
· `society` (a panel of 10 PARA personas, aggregated — the method) · `reward_only` (oracle upper
bound that maximizes the objective directly).

**Baked-in defaults** (you do **not** pass these — they're already the defaults):
- **10 steps**, **100 images** (50 PARA + 50 EVA, low-to-mid rated), **K=3** candidates, editor **FLUX.1-Kontext**.
- **Visible-edit emphasis** appended to every editor prompt (*"Make this a clearly visible edit, not a subtle one…"*) + **guidance_scale 3.0**, so edits are pronounced rather than near-invisible.
- **Drift cap 0.78** (loosened from 0.85 so the bolder edits actually commit while still guarding identity).

---

# Part C — Exactly what to run

## C.1 Smoke test first (2 images, ~2 min) — always

Loads every model and writes a few edits, so any failure surfaces immediately:

```bash
python script/c4_refine.py --dataset eva --n-images 2 \
    --conditions static,blind,society --editor flux --steps 3 --candidates 2 \
    --output-root /scratch/$USER/c4_smoke
```

Expect: models load → `[static]/[blind]/[society] … checkpoint` lines → `Done.`, with PNGs under
`/scratch/$USER/c4_smoke/edits/`. Glance at a couple of `edits/society/<id>/step*_best.png` to
confirm edits are visibly changing. If this works, the full run will too.

## C.2 The full experiment (this is the run to send)

```bash
OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh
```

That's it. `run_c4.sh` uses the target defaults (10 steps, 100 images, all 4 conditions, K=3, FLUX,
emphasis on, guidance 3.0, drift-cap 0.78), **auto-detects GPUs and shards the images across all of
them** (one shard per GPU, pinned via `CUDA_VISIBLE_DEVICES`), is **`--resume`-safe** (re-run the
same command to continue after a crash/timeout), and then generates the deliverable figures + table.

**For a long run, detach it** so it survives disconnects:
```bash
nohup env OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh > c4_run1.out 2>&1 &
tail -f c4_run1.out      # watch progress
```
(or drop the same command in an `sbatch` script — resumable, so requeue is safe.)

**Optional overrides** (only if needed):
```bash
NGPU=4            OUTPUT_ROOT=… scripts/run_c4.sh   # force 4-way shard (else auto)
TOTAL_IMAGES=200  OUTPUT_ROOT=… scripts/run_c4.sh   # bigger run
CONDITIONS=static,blind,society OUTPUT_ROOT=… scripts/run_c4.sh   # drop the oracle
EXTRA_ARGS="--drift-cap 0.80"   OUTPUT_ROOT=… scripts/run_c4.sh   # tighten identity guard
RUN_ANALYSIS=0    OUTPUT_ROOT=… scripts/run_c4.sh   # skip figures (make them later, C.3)
```

## C.3 (Re)generate the figures/tables anytime

Runs automatically at the end of C.2; to redo them (e.g. after copying the folder elsewhere):
```bash
cd scripts/analysis
python c4_trajectory.py  --output-root /scratch/$USER/c4_run1
python c4_qualitative.py --output-root /scratch/$USER/c4_run1
```

**Compute expectation:** 100 images × 4 conditions × 10 steps × 3 candidates ≈ **12k FLUX edits** +
persona critiques. ~a few hours on one H100; divide by ~N across N GPUs (4 → well under an hour).
`society` and `reward_only` are heaviest.

---

# Part D — Outputs (logs + images)

Everything lands under `OUTPUT_ROOT`:

```
$OUTPUT_ROOT/
  edits/<condition>/<image_id>/
      step0_source.png, step*_cand*.png, step*_best.png   # source + every candidate + committed best
  logs/c4_<condition>/
      c4_<condition>*.json                                # summary (config, panel, drift_cap)
      c4_<condition>*.part-*.json                         # per-step records (sharded, resumable)
  analysis/
      c4.json               # all metrics (gains, AUC, win-rates, convergence, drift, diversity)
      c4_summary.md         # the main results table
      figs/
        c4_trajectory.png     # best-so-far aesthetic vs step (the headline)
        c4_raw_objective.png  # RAW per-step objective (the fluctuating curve)
        c4_headline.png       # mean gain per condition + gain-vs-drift scatter
        c4_drift.png          # identity retention vs step (guardrail; cap line auto-matches the run)
        c4_diversity.png      # distinct complaints/step: society vs blind
        c4_qualitative.png    # source vs best-edit-per-condition grid
  stdout/                    # per-GPU run logs
```

---

# Part E — Zip and send

The whole run is self-contained under `OUTPUT_ROOT`. Zip it and transfer manually.

**Everything (logs + all images + analysis)** — biggest, has every edited PNG:
```bash
cd /scratch/$USER
zip -r c4_run1_full.zip c4_run1
# (tar is fine too:  tar -czf c4_run1_full.tar.gz c4_run1 )
```

**Lighter (logs + analysis only, no per-step images)** — small, enough to reproduce every number/figure:
```bash
cd /scratch/$USER
zip -r c4_run1_logs.zip c4_run1/logs c4_run1/analysis
```

**Just the best images per condition** (skip the intermediate candidates) if you want visuals but not GBs:
```bash
cd /scratch/$USER
zip -r c4_run1_bests.zip c4_run1/analysis $(find c4_run1/edits -name 'step*_best.png' -o -name 'step0_source.png')
```

Send `c4_run1_full.zip` (or the lighter ones) over your usual channel. On the receiving side it
unzips to the same `c4_run1/` layout, and `scripts/analysis/c4_trajectory.py --output-root c4_run1`
re-creates the figures from the logs.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `403` / gated repo when loading FLUX | Accept the `FLUX.1-Kontext-dev` license on HF; ensure `HF_TOKEN` is exported. |
| `HF_TOKEN is not set` during fetch | `export HF_TOKEN=…`; confirm access to `savoji/PARA`, `savoji/EVA`. |
| CUDA OOM | one 80 GB GPU per shard (don't over-subscribe); lower `--candidates`; or `EXTRA_ARGS=--cpu-offload` (slower). |
| torch can't see CUDA / wrong CUDA | `TORCH_INDEX_URL=…cuXXX scripts/setup_c4.sh`, or `SKIP_TORCH=1` after `module load`. |
| Run died partway | Re-run the **same** `run_c4.sh` command — `--resume` skips finished images (logs + per-shard cache persist). |
| One shard failed, others ran | Check `$OUTPUT_ROOT/stdout/c4_gpu*.log`; re-running resumes only the missing work. |
| Edits look too subtle | already mitigated (emphasis + guidance 3.0); if still timid, `EXTRA_ARGS="--drift-cap 0.72"` lets bolder edits commit. |
| Images drift into a different scene | tighten: `EXTRA_ARGS="--drift-cap 0.82"`. |
| `No c4 logs found …` from analysis | point `--output-root` at the folder that has `logs/c4_*`. |

## What "success" looks like

In `analysis/c4_summary.md`: **society** has the highest mean final gain and a **win-rate over blind
and static > 50%** (ideally the AUC CI excludes 0), while `c4_drift.png` shows committed edits staying
above the **0.78** cap — i.e. society *improves* images rather than just *transforming* them.
`c4_raw_objective.png` should visibly fluctuate (real edits happening), and `c4_trajectory.png` should
rise. Eyeball a few `c4_qualitative.png` panels: the win is real only if the society column looks
*better*, not merely *different*.
