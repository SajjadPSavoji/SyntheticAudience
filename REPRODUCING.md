# Reproducing AutoPolish

Three tiers. Each is self-contained — pick the deepest one your hardware and data access allow.

| Tier | Needs | Wall clock | Reproduces |
|---|---|---|---|
| [**1**](#tier-1--paper-figures-and-tables-cpu-minutes) | CPU + `savoji/AUTOPOLISH` | minutes | every number and figure in the paper |
| [**2**](#tier-2--re-analysis-from-raw-logs-cpu-1-hour) | CPU + `savoji/AUTOPOLISH` | ~1 hour | the analysis JSONs themselves, from raw logs |
| [**3**](#tier-3--full-runs-from-scratch-gpu-days) | GPUs + source datasets | days | the raw logs, from scratch |

Every result in the paper is seeded (`seed 0`) and every headline number carries a 1000-sample
bootstrap CI clustered by rater and image.

---

## Setup (all tiers)

```bash
git clone <this repo> && cd SyntheticAudience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set HF_TOKEN=<your token>   (HF_OWNER=savoji is already set)
```

You need **read access to the private `savoji/*` dataset repos**. The AutoPolish edits are
derivatives of the PARA and EVA photographs, so they are not public; request access from the
authors.

Verify your token before downloading gigabytes:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

---

## Tier 1 — paper figures and tables (CPU, minutes)

Fetch the cached analysis outputs, then rebuild every figure and the paper itself.

```bash
python scripts/fetch_from_hf.py autopolish
```

This restores two trees (~2.2 GB download; needs **~4.5 GB free** while running, because the
snapshot lands in `~/.cache/huggingface` before being copied into place):

- `results/` — 46 analysis JSONs and figures
- `data/results/` — 2,322 raw run logs and edited images

Reclaim the cache half afterwards with `huggingface-cli delete-cache`.

### Rebuild the figures

```bash
cd scripts/analysis

python paper_figs.py --c4-root ../../data/results/c4_run2   # main-text composite figures
python paper_extra.py                                       # supplement figures
```

Writes into `docs/paper/figs/`:

| File | Paper | Panels |
|---|---|---|
| `pf_audience.png` | Fig. 2 | calibration · between-group separation · panel-size curve |
| `pf_qualitative.png` | Fig. 3 | 3-row source/edit grid |
| `pf_autopolish.png` | Supp. | best-so-far trajectory · gain vs identity |

### Rebuild the paper

```bash
cd docs/paper && latexmk -pdf autoedit.tex
```

Expected: **19 pages** — 6 pages of main content, references starting on page 7, then the
supplement. No undefined references, no overfull boxes. Check with:

```bash
pdftotext autoedit.pdf - | awk 'BEGIN{p=1} /\f/{p++} {if ($0 ~ /^References$/) print "References: p"p}'
```

### Where each paper number lives

| Paper claim | Source |
|---|---|
| ICC(1)/ICC(k)/persona-ΔR² (§5.1) | `results/exp0.json` |
| Between-group separation (§5.2, Fig. 2b) | `results/c1_separation.json` |
| Calibration, group MAE vs prior (Fig. 2a) | `results/calibration.json` |
| Panel-size curve, crowd correlation (Fig. 2c) | `results/c3.json` |
| Steerability r = 0.37 / 0.39 | `results/steerability.json` |
| Rationale AUC 0.58–0.70 | `results/rationale.json` |
| AutoPolish table (§5.3) | `data/results/c4_run2/analysis/` |
| Holm correction (6 of 8) | `results/holm.json` |
| Subgroup fairness (0.50 / 0.23) | `results/bias.json` |
| Temperature robustness | `results/temp_compare.json` |

---

## Tier 2 — re-analysis from raw logs (CPU, ~1 hour)

Everything in `results/*.json` is re-derivable from `data/results/` with no GPU. This is the
tier that actually checks the analysis rather than trusting the cache.

```bash
cd scripts/analysis

# the four headline analyses
python exp0_ceiling.py        # variance decomposition / the ceiling
python calibration.py         # out-of-fold isotonic calibration
python c1_separation.py       # between-group separation (the headline)
python c3_rapidata.py         # generated-image transfer + panel-size curve

# supporting
python steerability.py rationale.py c2_ncurve.py holm.py temp_compare.py bias.py
```

To re-derive *everything* (33 scripts):

```bash
cd scripts/analysis
for f in *.py; do
  case "$f" in common.py|theme.py|attrs.py|paper_figs.py) continue;; esac
  echo "== $f"; python "$f" || echo "FAILED $f"
done
```

Each writes `results/<name>.json` (plus figures into `results/figs/`). Compare against the
fetched copies — they should match modulo float formatting.

### AutoPolish analyses

These read a run tree rather than `data/results/*` directly, so they take `--output-root`:

```bash
cd scripts/analysis
python c4_trajectory.py  --output-root ../../data/results/c4_run2
python c4_qualitative.py --output-root ../../data/results/c4_run2
python c4_progression.py --output-root ../../data/results/c4_run2
```

`c4_run2` is the run reported in the paper. `c4_run1` is an earlier pass kept for comparison.

---

## Tier 3 — full runs from scratch (GPU, days)

### Source data

```bash
python scripts/fetch_from_hf.py para eva lapis
```

Local layout after fetch: `data/para/imgs/<session>/<name>.jpg`,
`data/eva/images/<id>.jpg`, `data/lapis/images/<name>.jpg`, with ratings under each dataset's
`annotation/` or `data/` subdirectory.

### GPU environment

```bash
scripts/setup_c4.sh          # installs torch + requirements-gpu.txt, downloads model weights
```

Models pulled: `Qwen/Qwen2-VL-7B-Instruct` (the judge and critic),
`black-forest-labs/FLUX.1-Kontext-dev` (the editor), a LAION aesthetic predictor
(CLIP + MLP head, the held-out objective), and `facebook/dinov2-base` (the identity guardrail).
FLUX.1-Kontext-dev is gated — accept its license on the Hub first.

### Persona judge runs

These produce `data/results/{para,eva,lapis}_{full,blind}` — the input to Tiers 1 and 2.

```bash
python script/para_pipeline.py  --n-images 2000 --seed 0
python script/eva_pipeline.py   --n-images 1000 --seed 0
python script/lapis_pipeline.py --n-images 4000 --seed 0

# the persona-blind control (note the temperature — see below)
python script/para_pipeline.py --n-images 2000 --seed 0 --persona-blind --temperature 0.7
```

> **Always pair `--persona-blind` with `--temperature > 0`.** Under greedy decoding every rater
> of an image gets a byte-identical answer and the per-image distribution collapses to a point
> mass. This is the same decoding-collapse effect behind the paper's flat panel-size curve on
> real images (§6, limitation 1).

The temperature-0.7 arms reported in the supplement are the same commands with
`--temperature 0.7`, writing to `*_full_t07`.

`--analyze-only <log.json>` recomputes every metric from an existing log **without loading a
model** — use it to iterate on analysis without burning GPU time.

### The AutoPolish loop

```bash
OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh
```

Defaults match the paper: 10 steps, 100 images (50 PARA + 50 EVA), 4 conditions
(`static,blind,society,reward_only`), 3 candidates per step, FLUX editor, drift cap 0.78,
panel size 10. It auto-detects GPUs and shards the image set one shard per GPU. Overrides:
`STEPS`, `TOTAL_IMAGES`, `DATASET`, `CONDITIONS`, `CANDIDATES`, `NGPU`, `RUN_ANALYSIS`.

Full node-level runbook, including a smoke test and the shard/resume story:
[`docs/RUNNING_ON_H100.md`](docs/RUNNING_ON_H100.md).

Or run it directly:

```bash
python script/c4_refine.py --dataset both --n-images 50 --steps 10 --candidates 3 \
    --conditions static,blind,society,reward_only --editor flux \
    --drift-cap 0.78 --panel-size 10 --seed 0 --output-root /scratch/$USER/c4_run1
```

`--resume` continues an interrupted run; `--shard i/N` splits across processes.

### Cross-cultural preference on generated images (the C3 result)

`script/rapidata_pipeline.py` is the pairwise sibling of the three dataset pipelines: instead of
one persona scoring one image, it replays a real preference voter — reconstructed from the only
attributes Rapidata ships, **country and language** — choosing between two AI-generated images.
It reads `data/rapidata_700k/data/train_*.parquet` directly; there is no separate prep step.

```bash
# the paper's arm: 4 GPUs over disjoint pairs, then merge the shard logs
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python script/rapidata_pipeline.py \
    --n-pairs 800 --shard $i/4 --temperature 0.7 \
    --output data/logs/rapidata_run.json & done; wait

python script/rapidata_pipeline.py --analyze-only \
    "$(ls -m data/logs/rapidata_run.shard*of4.json | tr -d ' \n')"

# the persona-blind control
python script/rapidata_pipeline.py --n-pairs 800 --persona-blind --temperature 0.7
```

Three defaults encode design decisions worth knowing before you change them:

- `--criterion preference` (default) **does not** show the generation prompt, because the human
  annotators were prompt-blind too. `--criterion alignment` shows it and asks Rapidata's
  alignment question instead — a different ground truth, in a different Rapidata dataset.
- A persona here is a **stratum** (country × language), not an individual: the corpus has no
  stable rater IDs, so "replay rater #7" is not a thing this data supports.
- `--order random` counters position bias; `--order both` runs each pair in both orders.
  `--backend qwen` is required — LLaVA's template has one image slot and now refuses a pair
  rather than silently dropping one.

### The self-refine ablation

`script/claim4_pipeline.py` is the earlier, simpler loop reported as the self-refine comparison:
one *fixed* persona per image, edits chained on the previous output (not anchored), using
Qwen-Image-Edit rather than FLUX. It produced `data/results/c4_selfrefine/`.

```bash
python script/claim4_pipeline.py --n-images 100 --n-iterations 10 --seed 0
```

Contrast with `c4_refine.py`, the paper's system: a *panel* rather than one persona, an
**anchored** re-edit against the original image, and a **held-out** judge from a different model
family. Those three differences are what make AutoPolish non-circular.

### Unlabeled corpora

`script/pixel_pipeline.py` and `script/webscrap_pipeline.py` run the same persona judge over
image corpora that have **no** ground-truth ratings (`data/pixel/`, `data/webscrap/`). They
produce panel distributions with nothing to score against, so they support qualitative and
distribution-shape analysis rather than fidelity metrics. Not used in the paper.

---

## Publishing your own results

`scripts/push_to_hf.py` mirrors the fetch:

```bash
python scripts/push_to_hf.py autopolish            # results/ + data/results/ -> savoji/AUTOPOLISH
python scripts/push_to_hf.py autopolish --resume   # keep the repo, re-upload only what changed
```

Add a new dataset with one `DatasetSpec` line in `scripts/hf_dataset.py`. Two shapes are
supported: **corpus** (`images_dir` set — images become a parquet config browsable in the Hub
viewer) and **verbatim** (`images_dir=None` — the file tree is uploaded as-is, which is what
output trees like AutoPolish need, since the analysis scripts read them by path).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HF_TOKEN is not set` | Copy `.env.example` to `.env` and fill in the token. |
| `401` / `Repository Not Found` on fetch | The `savoji/*` repos are private; you need to be granted read access. |
| Fetch fills the disk | The snapshot is cached *and* copied — budget ~2× the dataset size. `huggingface-cli delete-cache` afterwards. |
| `no summary shards for run <name>` | `data/results/<run>/` is missing. Run `scripts/fetch_from_hf.py autopolish`. |
| Analysis figures look tiny in the PDF | Regenerate with `paper_figs.py`, which draws at final print width. Do not rescale its output in LaTeX. |
| Every persona returns the same score | Greedy decoding collapse. Pass `--temperature 0.7`. This is a known backbone property (~88% mode reversion on discrete rating tasks), not a bug. |
| FLUX download fails | `FLUX.1-Kontext-dev` is gated — accept the license on the Hub with the same account as `HF_TOKEN`. |
| OOM in the AutoPolish loop | Pass `--cpu-offload`, or reduce `--candidates`. The full loop wants ~40 GB VRAM. |
