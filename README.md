# AutoPolish: Improving Visual Aesthetics with Synthetic Audience Critique

*A frozen vision-language model role-plays a panel of viewers, their aggregated reaction
predicts how a real group will respond to an image, and that panel then drives an autonomous
image-editing loop.*

📄 **Paper:** [`docs/paper/neurips_creative_ai/autopolish.pdf`](docs/paper/neurips_creative_ai/autopolish.pdf) (6 pages + supplement)
· source: [`docs/paper/neurips_creative_ai/autopolish.tex`](docs/paper/neurips_creative_ai/autopolish.tex)
🎮 **Try it:** [`notebook/autopolish_playground.ipynb`](notebook/autopolish_playground.ipynb)
🔁 **Reproduce:** [`REPRODUCING.md`](REPRODUCING.md)

---

## The idea in one paragraph

Ask ten people to rate a photograph and you rarely get one answer. That disagreement is usually
averaged away as noise; we treat it as the signal. The honest answer to *"how will this image
land?"* is the **shape** of a population's reaction — its center, its spread, and how it shifts
between groups of viewers. We prompt a **frozen, off-the-shelf** VLM to role-play one specific
viewer at a time, described only in plain text (age, education, art familiarity, personality,
nationality), run it once per persona, and aggregate. **No weights are ever trained.** That
"panel of personas, then aggregate" step is the entire predictor.

One measurement organizes everything: an individual rating sits near the noise ceiling, so
predicting it is hard *by construction*, while the group aggregate is predictable precisely
because the idiosyncratic part of each reaction cancels in the average.

---

## Headline results

| | Result | Where |
|---|---|---|
| **Ceiling** | A single human rating shares only **19–47%** of its variance with other raters (ICC(1) 0.470 / 0.223 / 0.188 on PARA / EVA / LAPIS). The group mean is reliable to **ICC(k) 0.84–0.96**. Persona attributes explain just **1–4%** of individual taste. | paper §5.1 |
| **Group differences** | On paintings the persona panel recovers real between-group divergence at **r = +0.166** (95% CI [0.150, 0.181]) against a persona-blind control at +0.016. Signal tracks persona richness: LAPIS > PARA > EVA. | paper §5.2 |
| **Transfer to generated images** | On 23,445 crowd votes over 898 AI-generated pairs, panel-vs-crowd correlation **+0.466**, accuracy **0.659** vs a 0.600 majority prior, and accuracy rises monotonically with panel size (**0.588 → 0.657** for N = 1 → 20). | paper §5.2 |
| **AutoPolish** | A frozen synthetic-audience critic drives a non-circular, identity-preserving editing loop: held-out aesthetic gain **+0.170** (95% CI [0.136, 0.207]), identity 0.951, and **3.1×** richer feedback than a generic critic (3.12 vs 1.00 distinct complaints/step). | paper §5.3 |

**Reported negatives** (paper §6, in full): the panel-size curve is flat on *real* images
(greedy decoding collapses the panel); cross-country differentiation on generated images is null
and under-powered; the critic ablation ties on a *generic* aesthetic objective; thin personas
(EVA) break persona claims; and the judge carries a nationality/region calibration bias (gaps
0.50 / 0.23) that is measured and controlled rather than passed along.

---

## How AutoPolish works

```
  Image ─┐
         ├─> Frozen VLM ×N personas ─> aggregate + calibrate ─> group reaction
Persona ─┘                                                       (mean, spread, complaints)
 cards                                                                     │
                                                                complaints steer the critic
                                                                           ▼
  current best ──> audience critic ──> image editor ──> held-out judge ──> commit if better
     image          (complaints        (FLUX.1-        (LAION aesthetic     (loop ×10,
                    -> instruction)     Kontext)        + DINOv2 identity)   anchored re-edit)
```

The party that **proposes** edits is never the party that **grades** them — different model
families, and the grader never sees the instruction. That is what makes the loop non-circular.

---

## Quickstart

```bash
git clone <this repo> && cd SyntheticAudience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU: analysis + data sync

# zero-data smoke test (needs the GPU env below)
python script/opinions.py
```

Two environments, on purpose:

| Env | Install | Runs |
|---|---|---|
| **Analysis** (CPU) | `requirements.txt` | the 33-script re-analysis suite, figure generation, HF sync. `import editor` works without a GPU stack (imports are deferred). |
| **Inference** (GPU) | `requirements-gpu.txt` + torch, or `scripts/setup_c4.sh` | the VLM judge, FLUX editor, the full AutoPolish loop. ~40 GB VRAM for the full loop; the notebook's demos fit in ~16 GB. |

Start with [`notebook/autopolish_playground.ipynb`](notebook/autopolish_playground.ipynb) — it
runs a persona panel and one AutoPolish step on example images, and needs no private data.

---

## Repository layout

> **Note the two directories.** `script/` holds **experiment drivers** (things that produce
> results); `scripts/` holds **tooling** (things that move or analyze results). The names are
> historical and every path in `research_plan.md` depends on them, so they were left as-is.

```
docs/paper/            one self-contained directory per venue (copy the whole directory for a new one)
  neurips_creative_ai/ NeurIPS 2026 Creative AI: autopolish.tex + .pdf + checklist.tex + figs/
                       (rebuild: cd docs/paper/neurips_creative_ai && latexmk -pdf autopolish.tex;
                        figures: scripts/analysis/paper_figs.py --figs <venue>/figs)
REPRODUCING.md         step-by-step reproduction, tiered by what you have access to
research_plan.md       build plan, statistical endpoints, and the full interim-results log (§14)
PROPOSAL.md            the original scientific proposal
docs/architecture.md   how the code fits together + the non-obvious decisions

src/
  persona/             frozen VLM judge + persona role-play (Qwen2-VL / LLaVA backends)
  editor/              AutoPolish loop: critics, FLUX editor, aesthetic objective,
                       DINOv2 drift guardrail, refinement engine
  pipeline.py          generic "CSV of personas -> one image" demo

script/                EXPERIMENT DRIVERS
  {para,eva,lapis}_pipeline.py   replay real raters as VLM personas -> data/results/*
  rapidata_pipeline.py           PAIRWISE: replay real preference voters on generated-image
                                 pairs (the cross-cultural / generated-image transfer result)
  c4_refine.py                   the AutoPolish loop driver (the paper's headline system)
  claim4_pipeline.py             the earlier single-persona self-refine ablation
  {pixel,webscrap}_pipeline.py   judge unlabeled image corpora (no ground truth)
  opinions.py                    3-persona smoke test, no dataset needed
  export_results.py              stitch a sharded run log into one JSON

scripts/               TOOLING
  analysis/                      33-script re-analysis suite -> results/*.json + results/figs/
  analysis/paper_figs.py         main-text composite figures (print-size type)
  analysis/theme.py              the shared figure palette
  {push,fetch}_from_hf.py        sync data/ and results/ to private HF dataset repos
  setup_c4.sh / run_c4.sh        one-command GPU setup + full AutoPolish run

notebook/
  autopolish_playground.ipynb    START HERE — panel + one edit step + headline numbers
  c4_colab.ipynb                 Colab runner for a full AutoPolish run (A100)
  *_EDA.ipynb                    per-dataset exploratory analysis
  *_results.ipynb                per-dataset run results
  *_vs_groundtruth.ipynb         persona predictions against the real ratings

data/                  git-ignored — datasets + raw run outputs, fetched from HF
results/               git-ignored — analysis JSON + figures, fetched from HF
```

---

## Data and artifacts

Nothing large or license-restricted is committed. Everything lives in **private Hugging Face
dataset repos** and is rehydrated on demand:

| Repo | What | Restores to |
|---|---|---|
| `savoji/PARA` | 4,000 photos, 1–5 aesthetic + 8 sub-axes, Big-Five personas | `data/para/` |
| `savoji/EVA` | 4,070 photos (AVA subset), 0–10 score + 4 attributes | `data/eva/` |
| `savoji/LAPIS` | 4,000 paintings, 0–100 rating, nationality | `data/lapis/` |
| `savoji/AUTOPOLISH` | **all experiment outputs** — 46 analysis JSONs/figures + 2,322 raw run logs and edited images (~2.2 GB) | `results/` + `data/results/` |

```bash
cp .env.example .env          # then set HF_TOKEN (HF_OWNER=savoji is already there)
python scripts/fetch_from_hf.py autopolish      # results only  (~2.2 GB, needs ~4.5 GB free)
python scripts/fetch_from_hf.py para eva lapis  # source images (large)
```

These repos are **private** — the AutoPolish edits are derivatives of the PARA and EVA
photographs and inherit their terms. You need to be granted read access.

---

## Reproducing

Three tiers, cheapest first. Full detail in [`REPRODUCING.md`](REPRODUCING.md).

| Tier | Needs | Cost | Reproduces |
|---|---|---|---|
| **1 — Figures + tables** | CPU, `savoji/AUTOPOLISH` | minutes | every number and figure in the paper, from cached analysis JSONs |
| **2 — Re-analysis** | CPU, `savoji/AUTOPOLISH` | ~1 hour | those JSONs themselves, re-derived from the raw run logs |
| **3 — Full runs** | 1–8 GPUs, source datasets | days | the raw logs, from scratch: judge runs + the AutoPolish loop |

All three tiers are complete: every experiment in the paper has its generating script in
`script/`, including the Rapidata cross-cultural run and the self-refine ablation.

---

## Citation

The paper is under anonymous review; please cite the arXiv version when it is posted. Until
then, refer to `docs/paper/neurips_creative_ai/autopolish.pdf`.

## License

Code in this repository is available for research use. The datasets it consumes (PARA, EVA,
LAPIS, Rapidata) carry their own licenses and are **not** redistributed here — see
[`docs/para.md`](docs/para.md), [`docs/eva.md`](docs/eva.md), and [`docs/lapis.md`](docs/lapis.md)
for each dataset's source and terms.
