# Synthetic Audiences for Creative AI

*Persona-conditioned multimodal judges that predict human reactions and guide image editing.*

When we ask "is this image good?" we are not asking for a single number — appeal is
**collective**, a distribution of reactions across viewers who differ in culture, training,
and taste. This project simulates that collective reaction with an **off-the-shelf, frozen**
vision-language model (VLM): we prompt it to role-play many viewers ("personas"), run it once
per persona, and **aggregate** the reactions into a predicted *group* reaction. That
panel-of-personas-then-aggregate step is the whole method — no VLM is ever trained or
fine-tuned.

> **Model note.** The headline backbone is a **local, frozen `Qwen/Qwen2-VL-7B-Instruct`**.
> The original proposal called models "serverlessly"; that requirement was dropped (a *frozen*
> model is the claim — local vs. hosted is an implementation detail). See `research_plan.md`
> §14.8.

Full scientific write-up: [`PROPOSAL.md`](PROPOSAL.md). Step-by-step plan, interim results, and
the running experiment log: [`research_plan.md`](research_plan.md).

---

## The four claims

| | Claim | Status |
|---|---|---|
| **Exp 0** | Individual taste sits near the noise ceiling; the *group mean* is highly reliable (ICC(1) 0.19–0.47 vs ICC(k) 0.84–0.96). This motivates aggregation. | ✅ proven |
| **C1** | **(headline)** A panel of personas, aggregated, reproduces a group's reaction — and crucially the *differences between groups* — better than a no-persona judge or the population-mean prior. | 🟡 demonstrated on LAPIS (nationality/age/art-interest), weak on PARA, fails on EVA |
| **C2** | *Why* C1 works: individual error ≫ group error, and error falls then saturates as the panel grows (the N-curve). Calibrated group prediction beats the population prior on all datasets. | 🟡 mean/rank proven; the aggregation-mechanism N-curve still needs a decoding fix |
| **C3** | It generalizes zero-shot from real images to **AI-generated** ones, across cultures (Rapidata per-country votes). | ⚪ active workstream |
| **C4** | The simulated audience is a **useful, non-circular editing signal**: a *society* of personas gives better edit feedback than a single blind VLM or a fixed instruction, inside an auto-refinement loop. | 🟢 implemented + smoke-tested (see below) |

C4 was reframed (2026-07-15, `research_plan.md` §14.19) into a **demographic-free critic-quality
ablation** inside a 10-step auto-refinement editing loop. That is the code in `src/editor/` +
`script/c4_refine.py` and the focus of the runbook below.

---

## Repository layout

```
PROPOSAL.md            scientific proposal (idea, claims, positioning)
research_plan.md       the build plan + interim results + running log (source of truth)
README.md              this file
requirements.txt       minimal analysis/data env (no GPU stack)
requirements-gpu.txt   heavy GPU stack for inference + C4 (torch installed separately)

src/
  persona/             frozen VLM judge + persona role-play  (Qwen2-VL / LLaVA backends)
  editor/              C4 auto-refinement loop: editor (FLUX), aesthetic objective,
                       drift guardrail, critics (static/blind/society), loop engine
script/
  {para,eva,lapis}_pipeline.py   replay real raters as VLM personas -> data/results/*
  c4_refine.py                   C4 auto-refinement driver (the editing loop)
scripts/
  fetch_from_hf.py / push_to_hf.py   sync data/ to/from private HF dataset repos
  setup_c4.sh / run_c4.sh            one-command GPU setup + full C4 run  (see RUNNING_ON_H100.md)
  pull_c4_from_drive.sh             pull a C4 run from Google Drive via rclone
  analysis/                         re-analysis suite (Exp0, C1, C2, calibration, C4 figures, ...)
data/                  git-ignored; datasets + run outputs, fetched from HF
results/               git-ignored; analysis JSON + figures
notebook/c4_colab.ipynb            Colab runner for C4 (A100)
docs/                  per-dataset notes, protocols, claim specs
```

`data/` and `results/` are **git-ignored** — nothing large or license-restricted is committed.

---

## Data

Three real-image rating datasets (each image rated by ~25–35 real people) plus generated-image
preferences for C3:

| Dataset | What | Used for | Source |
|---|---|---|---|
| **PARA** | photos, 1–5 aesthetic + 8 sub-axes, Big-Five personas | Exp0, C1, C2, C4 personas | private HF `savoji/PARA` |
| **EVA** | photos (AVA subset), 0–10 score + 4 attributes | Exp0, C1, C2, C4 sources | private HF `savoji/EVA` (CC0) |
| **LAPIS** | paintings, 0–100 rating, nationality | Exp0, C1, C2, C3 precursor | private HF `savoji/LAPIS` |
| **Rapidata** | per-country votes on generated images | C3 | public HF (pulled) |

Datasets are stored in **private Hugging Face dataset repos** and rehydrated into `data/` with:

```bash
python scripts/fetch_from_hf.py eva para lapis     # needs HF_TOKEN + access to the repos
```

Copy `.env.example` to `.env` and set `HF_TOKEN` (and `HF_OWNER=savoji`). You must have been
granted read access to the `savoji/*` dataset repos.

Local layout after fetch: `data/eva/images/<id>.jpg`, `data/para/imgs/<session>/<name>.jpg`,
`data/lapis/images/<name>.jpg`, with ratings under each dataset's `annotation/` or `data/`.

---

## Environments

There are two, on purpose:

- **Analysis / data env** (`requirements.txt`) — no GPU. Runs the re-analysis suite in
  `scripts/analysis/` and the data fetch. The `editor` package is import-safe here (heavy
  imports are deferred), so you can develop without a GPU.
- **GPU env** (`requirements-gpu.txt` + torch) — runs the VLM judge, FLUX editor, and the whole
  C4 loop. Set up with `scripts/setup_c4.sh` (see the H100 runbook).

---

## Running things

### Persona judge over a dataset (produces `data/results/*`)
```bash
python script/eva_pipeline.py  --n-images 1000 --output data/logs/eva_full.json
python script/para_pipeline.py --n-images 2000
python script/lapis_pipeline.py --n-images 4000
```
Each replays real raters as personas (`full`) or a no-persona control (`--persona-blind`) and
writes sharded JSON logs.

### Re-analysis suite (no GPU)
```bash
cd scripts/analysis
python exp0_ceiling.py      # Exp 0 variance decomposition / ceiling
python c1_separation.py     # C1 between-group separation (the headline)
python c2_ncurve.py         # C2 aggregate-vs-individual gap + N-curve
python calibration.py       # post-hoc calibration
# ... 28 scripts total; outputs -> results/*.json + results/figs/*.png
```
See `research_plan.md` §14 for what each one found.

### C4 — the auto-refinement editing loop (GPU)
The headline runnable of this repo. See **[`docs/RUNNING_ON_H100.md`](docs/RUNNING_ON_H100.md)**
for the full runbook. In short:
```bash
scripts/setup_c4.sh                                    # one-time env setup
OUTPUT_ROOT=/scratch/$USER/c4_run1 scripts/run_c4.sh   # 10 steps, 100 images, all GPUs
```
This runs three critic conditions (`static` / `blind` / `society`) + a `reward_only` oracle
through a 10-step **anchored-re-edit + accept-if-better** loop, then produces the deliverables:
best-so-far trajectory curves, a headline gain + drift-vs-gain figure, a summary table, and a
qualitative before/after grid under `$OUTPUT_ROOT/analysis/`.

---

## Key documents

- [`PROPOSAL.md`](PROPOSAL.md) — the idea, the four claims, related work.
- [`research_plan.md`](research_plan.md) — build plan, statistical endpoints, and the full
  interim-results log (§14). The current source of truth for status.
- [`docs/RUNNING_ON_H100.md`](docs/RUNNING_ON_H100.md) — coworker runbook for the GPU node.
- [`docs/claim3_cross_cultural.md`](docs/claim3_cross_cultural.md) — the active C3 spec.
