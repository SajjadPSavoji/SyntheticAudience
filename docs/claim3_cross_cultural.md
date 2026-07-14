# Claim 3 (C3): cross-cultural preference on AI-generated images

**Owner:** teammate (inference) + us (data-prep & analysis, local/no-GPU) ·
**Model:** the same frozen `Qwen/Qwen2-VL-7B-Instruct` · **Env:** the `persona` conda env,
`script/*_pipeline.py`.

This document is self-contained: the high-level idea, the exact data source, the experiment, the
code to add to the current repo, how to export results (same chunked-log format as before), and how
we analyze them locally afterward.

---

## 0. TL;DR

We already showed (on real art, LAPIS) that a frozen persona-conditioned VLM reproduces how
**different nationalities** rate the same painting differently (between-group separation +0.17). **C3
asks whether that same cross-cultural signal carries over to AI-generated images.** We take
**Rapidata** — 700k human preference votes over FLUX / SD3.5 / Midjourney / DALL·E 3 image pairs,
**with each voter's country recorded** — and test whether the judge, conditioned on a viewer's
country, predicts which image *that country* prefers, specifically on the pairs where countries
actually disagree.

You (teammate) build one new pipeline, `script/rapidata_pipeline.py`, that scores each generated
image's appeal from the point of view of a given country, and export chunked logs exactly like the
PARA/EVA/LAPIS runs. We do the pairwise analysis locally.

---

## 1. High-level idea

- **The claim.** The same zero-shot recipe — frozen VLM + a short natural-language "viewer profile" +
  post-hoc calibration fixed on real images — predicts **cross-cultural preferences on AI-generated
  images**, without touching any weights.
- **Why it's the right next step.** Our strongest real-image result is the LAPIS **nationality**
  separation. C3 is literally "does that nationality signal transfer from real paintings to generated
  images?" It reuses everything we built (persona card, calibration, metrics).
- **The two confounds C3 must kill** (this is the whole point of the design):
  1. **Universal quality** — some generated images are just better and *everyone* prefers them.
     Predict the winner from that and you look right without any cultural understanding. → we measure
     improvement *over a global-preference baseline*, and only on **disagreement pairs**.
  2. **Is the persona doing anything?** → a **no-persona control** (generic prompt), same as the
     `blind` runs.

---

## 2. Data source (Hugging Face)

Public dataset — you **pull** it, nothing to push.

| Axis | HF dataset id | Use |
|---|---|---|
| **Preference** (primary) | `Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3` | the C3 target |
| Coherence (optional) | `Rapidata/Flux_SD3_MJ_Dalle_Human_Coherence_Dataset` | secondary axis, optional |
| Alignment (optional) | `Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset` | secondary axis, optional |

**Row schema (verified):**
```
prompt            : str
image1, image2    : embedded images (PIL, ~1024x1024)   # the two candidates
votes_image1/2    : int                                  # total votes each
model1, model2    : str   e.g. "dalle-3", "flux", "sd3", "midjourney"
image1_path/2     : str   e.g. "dalle-3/155_0.jpg", "flux/155_0.jpg"
detailed_results  : str (JSON) = {"votes":[
                      {"votedFor":"flux/155_0.jpg",
                       "userDetails":{"country":"IN","language":"en","userScore":0.78}}, ... ]}
```
Key point: **per-vote `country` and `language` live inside `detailed_results`.** Countries are
ISO-3166 alpha-2 codes (`IN`, `GB`, `DZ`, …). The split is sharded: use `split="train_0001"` etc.
(there is no single `"train"`).

---

## 3. The experiment (what we compute)

**Unit of prediction:** for an image pair (A, B) and a country *c*, which image does country *c*
prefer? Observed from Rapidata: `winrate_A(c) = votes_A(c) / (votes_A(c)+votes_B(c))`.

**The judge's job:** produce a scalar **appeal score** for each *single* image, conditioned on a
country-viewer. Then the predicted pairwise preference is
`p̂(A ≻ B | c) = σ(β · (ŝ_A(c) − ŝ_B(c)))`, with β fit on a validation split (we do this locally).

**Crucial simplification vs. the real-image runs.** For C3 the "persona" is **only the country**
(Rapidata has no age/gender/Big-5). So every voter from country *c* gets the *same* viewer profile →
one judge call per **(image, country)** is the slice prediction. **There is no within-country panel
to aggregate**, so the temperature/mode-collapse problem that bit C1/C2 does **not** apply here — a
single deterministic score per (image, country) is exactly what we need. Use `--temperature 0`.

**Primary metric (computed locally):** on **disagreement pairs** (pairs where the top countries
genuinely split — `max_c winrate_A(c) − min_c winrate_A(c) > 0.2`, each slice ≥ 30 votes),
```
ΔAUC = AUC(slice-conditioned p̂)  −  AUC(global-preference baseline)
```
restricted to the **LAPIS-nationality subset**. Positive ΔAUC with a CI excluding 0 ⇒ the persona
adds real cross-cultural predictive power beyond universal quality. **Controls:** the global-
preference baseline (universal quality) and the no-persona run (`blind`).

---

## 4. What to build in the repo

Two new files under `script/`, mirroring the existing dataset pipelines. Reuse `para_pipeline.py`
machinery (`ScoreDimension`, JSON parsing, chunked-log IO, batching) exactly as `eva_pipeline.py`
and `lapis_pipeline.py` do.

### 4.1 `script/build_rapidata.py` — data prep (local, no GPU)

Downloads Rapidata and produces the inputs the judge and the analysis need. Steps:

1. **Load** all shards of the preference set (`load_dataset(..., split="train_0001")`, loop shards;
   or non-streaming if disk allows — it's a few GB).
2. **Explode `detailed_results`** into per-(pair, country) vote counts →
   `pairs.parquet` with columns:
   `pair_id, prompt, image_a_path, image_b_path, model_a, model_b, country, votes_a, votes_b, n_votes, winrate_a`.
   (`pair_id` = a stable hash of the two image paths.)
3. **Country filtering:** map ISO-2 → LAPIS nationality name (`GB→british`, `PL→polish`, …; we will
   supply `data/lapis_nationality_map.csv`). Keep only countries in the **LAPIS intersection**, then
   the **top-N by vote volume** (start N≈10–15). Keep slices with `n_votes ≥ 30`.
4. **Disagreement pairs:** flag pairs where `max−min winrate_a > 0.2` across the kept countries.
5. **Export the unique images to disk** so the judge can load them by path (same pattern as the other
   datasets — an `image_root` + relative path). Write each distinct `image_path` (e.g. `flux/155_0.jpg`)
   to `data/rapidata/images/<image_path>`. Also write `images_to_score.csv`
   (`image_path, model`) — the deduplicated set of images that appear in the kept disagreement pairs
   (this is all the judge needs to score; typically a few thousand, **not** 700k).
6. **Outputs** (under `data/rapidata/`): `images/…`, `images_to_score.csv`, `pairs.parquet`,
   `disagreement_pairs.parquet`, and a `countries.json` (the kept slice list). `data/` is git-ignored,
   so this stays local.

> We (local) can also run this step and hand you `images_to_score.csv` + the images if you prefer to
> only do the GPU part — your call.

### 4.2 `script/rapidata_pipeline.py` — the judge run (GPU, teammate)

A sibling of `eva_pipeline.py` / `lapis_pipeline.py`. Differences from those:

- **One scoring axis:** `appeal` on a **0–100** scale (same as LAPIS, so we can reuse the LAPIS
  calibration). Define it as a single `ScoreDimension("appeal", 0, 100, 1, "...")`.
- **Persona = country/language only.** The persona-card builder takes a country (+language) and emits
  e.g. *"You are a viewer from {country_name}. React to this image the way people from your country
  typically would."* Define `RAPIDATA_SYSTEM_PROMPT_TEMPLATE` (persona) and
  `RAPIDATA_GENERIC_SYSTEM_PROMPT` (no-persona control), and a `build_rapidata_question(persona_blind)`
  — copy the shape from `eva_pipeline.py`.
- **Task list = (image, country) pairs**, not (image, rater). Read `images_to_score.csv` and
  `countries.json`; the set of tasks is the cross product {images_to_score} × {kept countries}. In
  `--persona-blind` mode, tasks are just {images_to_score} once (generic prompt, no country).
- **Record schema per task:** `image_path, model, country, language, gt=None` (there is no per-image
  ground-truth score — the truth is the pairwise winrate, assembled later), plus
  `pred_appeal, comment, raw_response`. Keep the same chunked-summary + `.part-NNNN.json` writer from
  `para_pipeline.py`.
- **CLI:** keep the standard flags (`--model-name`, `--temperature`, `--persona-blind`, `--shard i/N`,
  `--chunk-size`, `--output`, `--resume`, `--batch-size`). Add `--countries` (subset) and
  `--image-list data/rapidata/images_to_score.csv` for convenience.
- **`image_root`:** `data/rapidata/images` (so `image_path` = `flux/155_0.jpg` resolves).
- **Temperature: `--temperature 0`** (see §3 — we only need the per-(image,country) point estimate;
  no panel, so no need for sampling here).

---

## 5. Run commands (what you execute)

```bash
# 1) data prep (CPU; ~minutes + a few GB download)
python script/build_rapidata.py --top-countries 12 --min-votes 30 --disagreement-threshold 0.2

# 2) persona (country-conditioned) run — the main condition
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/rapidata_pipeline.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct --temperature 0 \
    --image-list data/rapidata/images_to_score.csv \
    --shard $i/4 --chunk-size 5000 \
    --output data/logs/rapidata_full.shard${i}of4.json &
done; wait

# 3) no-persona control (generic prompt, one score per image)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/rapidata_pipeline.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct --temperature 0 --persona-blind \
    --image-list data/rapidata/images_to_score.csv \
    --shard $i/4 --chunk-size 5000 \
    --output data/logs/rapidata_blind.shard${i}of4.json &
done; wait
```

---

## 6. Export & hand-off (same convention as before)

- Name the runs **`rapidata_full`** (country-conditioned) and **`rapidata_blind`** (no-persona
  control), sharded exactly like the previous deliveries (per-shard summary `…shardIofN.json` +
  `…part-NNNN.json` chunks).
- Drop each run's files into its own folder and hand back:
  ```
  data/results/rapidata_full/…
  data/results/rapidata_blind/…
  ```
  plus the **`data/rapidata/` prep outputs** (`pairs.parquet`, `disagreement_pairs.parquet`,
  `countries.json`, `images_to_score.csv`) so we can run the analysis. `data/` is git-ignored — keep
  it local / hand it over directly (the folders are a few hundred MB).

---

## 7. What we do locally afterward (analysis — no action needed from you)

1. Join each `(image_path, country)` judge score to `pairs.parquet`; for each disagreement pair and
   country form `Δŝ = ŝ_A(c) − ŝ_B(c)`.
2. Fit `β` in `p̂(A≻B|c)=σ(β·Δŝ)` on a validation split; evaluate pairwise accuracy + **ΔAUC vs the
   global-preference baseline** on the held-out disagreement pairs, with bootstrap CIs.
3. Report the two controls (global-preference, no-persona) and the **real-to-generated gap** vs the
   LAPIS C1 result — reusing the frozen calibration (`§14.13.6` showed it transfers) and netting out
   the known national rating bias (`§14.12.5`).

---

## 8. Acceptance criteria / sanity checks

- `build_rapidata.py` prints: #pairs, #countries kept (LAPIS intersection), #disagreement pairs,
  #unique images to score. **Sanity:** ≥ ~8 countries and a few hundred disagreement pairs, else relax
  `--top-countries` / thresholds.
- Every `image_path` in `images_to_score.csv` exists under `data/rapidata/images/`.
- `rapidata_full` covers **{images} × {countries}** tasks; `rapidata_blind` covers **{images}** once.
- Parse rate ≈ 100% and `pred_appeal` spans the 0–100 range (not stuck at one value) — quick check
  before handing off.

---

## 9. Notes / rationale

- **Why one call per (image, country) and not a panel:** the slice is defined *only* by country, so
  the persona card is identical for all voters of a country → the panel would be a point mass anyway.
  We only need the slice-mean appeal to rank the pair. This side-steps the temperature/mode-collapse
  issue entirely — C3 does not depend on the fix we parked.
- **Why 0–100 appeal:** matches LAPIS, so the calibration we fit on LAPIS transfers directly (validated
  in §14.13.6), and the real-to-generated comparison is apples-to-apples.
- **Cost:** unique images in disagreement pairs (a few thousand) × (≈12 countries + 1 generic) — on
  the order of tens of thousands of judge calls, comparable to a single real-image run, not 700k.
- **Ethics:** frame outputs as dataset-sampled country distributions, not national essences; we report
  and net out the model's own national rating bias (§14.12.5).
