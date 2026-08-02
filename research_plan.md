# Synthetic Audiences for Creative AI — Research Plan

**Persona-conditioned multimodal judges that predict human reactions and guide image editing.**

*Consolidated plan + results. Last updated 2026-07-25. This document is the single source of truth
for the project's motivation, method, experimental design, and results to date. Run-by-run
chronological notes and every raw number live in `results/*.json`, `data/results/*`, and git
history; this file states the current, cross-checked conclusions.*

---

## Table of contents

1. Motivation and thesis
2. Claims and contributions (status at a glance)
3. Positioning relative to prior work
4. Data
5. Method
6. Experimental design and metrics
7. Statistical analysis plan
8. Results
9. Ablations
10. Limitations and honest negatives
11. Roadmap
12. Reproducibility, code, and artifacts
13. Ethics
- Appendices A–H

---

## 1. Motivation and thesis

Aesthetic judgment is **plural**. The same image delights one viewer and bores another, and both
reactions are valid — they belong to different people. So the honest target for "how good is this
image?" is not a scalar but the **shape of the reaction across a population**: its mean, its spread,
and how it shifts between groups of viewers.

We ask whether we can **simulate that collective reaction** with an off-the-shelf, **frozen**
vision-language model (VLM), and whether the simulated audience is useful enough to guide creative
work. The method is deliberately simple: prompt the VLM to role-play many viewers ("personas"),
described in plain text (age, education, art experience, personality, nationality); run it once per
persona; then **aggregate** the reactions into a predicted *group* distribution. This
panel-of-personas-then-aggregate step is the entire method — no model is trained or fine-tuned.

A single insight organizes everything: the unit of interest is the **group, not the individual**.
Individual taste sits near the noise ceiling and is barely predictable; the group aggregate is
predictable because the idiosyncratic part of each reaction averages away. Making this contrast
explicit — individual prediction is hard *by design*, group prediction is where the signal lives —
is a contribution, not a limitation to hide.

**Frozen-model note.** The headline backbone is a **local, frozen `Qwen/Qwen2-VL-7B-Instruct`**. The
original proposal required *serverless* calls; that requirement was dropped — a *frozen* model is the
scientific claim, and local vs. hosted is an implementation detail. Only two kinds of fitting are
ever used, and neither touches VLM weights: (a) post-hoc score **calibration**, and (b) classical
**non-VLM baselines** (a regressor and collaborative filtering).

---

## 2. Claims and contributions (status at a glance)

| | Claim | Status | Decisive evidence |
|---|---|---|---|
| **Exp 0** | Individual taste is near-noise; the group mean is highly reliable — motivating aggregation. | ✅ **proven** | ICC(1) 0.19–0.47 vs ICC(k) 0.84–0.96; persona-ΔR² only 1–4% (§8.1) |
| **C1** | A persona panel reproduces a group's reaction, and specifically the **differences between groups**, better than a no-persona judge or the population-mean prior. | 🟢 **demonstrated on LAPIS** (weak on PARA, fails on EVA) | between-group separation **+0.17 [0.15, 0.18]** vs blind ≈0; Holm-significant; temperature-robust (§8.4) |
| **C2** | *Why* C1 works: individual error ≫ group error, and calibrated aggregates beat the prior. | 🟡 **mean/rank proven; aggregation-mechanism (N-curve) not shown on real images** | calibrated group MAE beats prior on all 16 axes; but the N-curve is flat under greedy decoding (§8.3, §8.9) |
| **C3** | The method generalizes to **AI-generated** images. | 🟢 **C3a (aggregation) supported** / 🔴 C3b (cross-cultural) null | aggregate r=+0.47, panel 0.66 > 0.60 majority, clean N-curve 0.59→0.66; between-country separation +0.00 (§8.6) |
| **C4** | The simulated audience is a **usable, non-circular editing signal**. | 🟢 **premise validated** / 🟡 critic-ablation null on a generic objective | held-out aesthetic gain CIs all exclude 0 (society +0.170), identity preserved, society feedback 3.1× richer (§8.7) |

**One-paragraph summary.** The supporting science is in hand: individual taste is near-noise while
the group is reliable (Exp 0), the persona is provably functional (steerability), and calibrated
aggregates beat the population prior everywhere. The headline **C1** between-group result is
demonstrated on art (LAPIS), including the nationality axis. On generated images (**C3**) the
*aggregation mechanism* transfers cleanly (the first rising N-curve in the project) even though
cross-cultural *differentiation* does not. The editing loop (**C4**) is validated as a non-circular,
identity-preserving signal. Two honest negatives — the flat N-curve on real images (a decoding
limitation) and the null critic-ablation on a generic aesthetic objective — are reported in full and
each points to a concrete next experiment.

---

## 3. Positioning relative to prior work

- **PAMELA (2026)** personalizes a scalar reward to a *single* user and optimizes prompts against it.
  We instead model a *population distribution*, keep the generator frozen, and feed back
  natural-language critique rather than a scalar.
- **AesBiasBench (2025)** provides the lens we use to check subgroup calibration (see §8.8).
- **Individual-differences-in-computational-aesthetics (2025)** motivates our explicit "ceiling"
  analysis (Exp 0).
- Unlike text-only "silicon sampling," our audience reacts to *visual* stimuli.

This work is *not* another personalized-aesthetics model: we do not train a taste predictor. The
novelty is (a) predicting a *group* rather than one user, (b) making aggregation the measured
mechanism, and (c) using the audience as interpretable creative feedback.

---

## 4. Data

### 4.1 Datasets

| Dataset | Content | Scale used | Native scale | Persona fields | Role |
|---|---|---|---|---|---|
| **PARA** | photos | 4,000 imgs / ~52k ratings (of 31k/808k) | 1–5 aesthetic + 8 sub-axes | age, gender, education, art & photo experience, Big-Five | Exp0, C1, C2, C4 source & personas |
| **EVA** (CC0) | photos (AVA subset) | 100% (4,070 / 137k) | 0–10 + 4 attribute votes | age, gender, region, photographic level | Exp0, C1, C2, C4 source |
| **LAPIS** | paintings | 4,000 imgs / 90k ratings (34%) | 0–100 | nationality, art interest, age | Exp0, C1, C2; C3 precursor |
| **Rapidata** | generated-image pairs (FLUX/SD3/MJ/DALLE3) | 23k votes / 898 pairs / 120 countries | pairwise vote | per-vote country + language | C3 |
| **EditReward-Bench** | edit-preference pairs | reference | — | — | (dropped as the C4 objective — see §5.5) |

Datasets live in **private HF dataset repos** (`savoji/PARA`, `savoji/EVA`, `savoji/LAPIS`) and are
rehydrated into `data/` with `scripts/fetch_from_hf.py`; Rapidata is public (pulled). `data/` and
`results/` are git-ignored.

### 4.2 Unified rating schema (`processed/ratings.parquet`)

One row per (image, rater), every score normalized to `score_norm ∈ [0,1]`
(PARA `(s−1)/4`, EVA `s/10`, LAPIS `s/100`). Fields absent in a dataset are null.

```
image_id, image_path, dataset {para,lapis,eva}, user_id, score_norm,
emotion, content_pref, willingness_share,            # PARA
difficulty, eva_attr_{visual,composition,quality,semantic},  # EVA / PARA
age_bucket, gender, education, art_exp, photo_exp,
big5_{O,C,E,A,N}, nationality, region                # persona attributes
```

*EVA fills only age/gender/region/photo_exp on the persona side (no Big-Five, education, art
familiarity, or nationality) — which is why it is the weakest dataset for persona-dependent claims
and does not enter the C3 cross-cultural slice.*

### 4.3 Splits

- **cold_start** (primary C1 regime): hold out a disjoint 20% of *users*; the model predicts for a
  viewer it has no data on, from the persona card alone.
- **warm_start**: same users in pool and test, holding out (user, image) pairs — the collaborative-
  filtering baseline lives here only.
- **image-disjoint** (and artist-disjoint for LAPIS) folds for leakage control.
- All split indices saved as JSON; global `seed = 0`.

### 4.4 Generated-image table (`processed/rapidata_pairs.parquet`)

One row per (pair, slice): `pair_id, prompt, image_a/b_path, model_a/b, slice_key (country|language),
votes_a, votes_b, n_votes, winrate_a`. Keep slices with `n_votes ≥ 30`; restrict to nationalities
also in LAPIS for the cross-cultural analysis.

---

## 5. Method

### 5.1 The persona card (`src/data/persona.py` / `build_para_description`)

A deterministic text template built from a user's attributes; missing fields are omitted, never
invented. Example:

> *Viewer profile: 35–44, master's degree, high art familiarity, moderate photography experience,
> nationality: Belgium. Personality (Big-5, 1–5): O 4.2, C 3.1, E 2.8, A 3.9, N 2.5.*

**Field-set consistency is critical:** the card must emit the *same fields* in the reference pool and
at test time for any attribute we claim generalization on. For Rapidata (C3) only country/language
are available, so the card there uses only those — which is why the cross-cultural claim is scoped to
nationality/country.

### 5.2 The frozen judge (`src/persona/backend/qwen.py`)

A single swappable class wrapping the frozen VLM. Input = image + persona card + output-format
instruction; output = JSON `{score, emotion, willingness, difficulty, rationale}` (Appendix A),
parsed with a tolerant parser (strict JSON → per-key regex fallback). Weights are never touched.
Default backbone: **Qwen2-VL-7B-Instruct**, batched, `sdpa` attention, bf16.

### 5.3 The synthetic audience (`src/audience/`)

- `sample_personas(target_distribution, N)` — draw N persona cards to match a target group.
- `predict_group(image, personas)` — run the judge once per persona → an empirical group distribution.
- `aggregate(distribution)` — mean, dispersion, top complaints (mined from rationales),
  between-group disagreement.
- `sweep_N` — recompute at N ∈ {1,2,5,10,20,50} for the C2 fidelity curve.
- **Caching** keyed by `(image_id, persona_hash, model_id, prompt_version, decoding)` — one query
  per (image, persona) feeds every panel, slice, and N-sweep.

### 5.4 Post-hoc calibration (`src/predict/calibrate.py`)

An isotonic map from raw to calibrated score, fit **out-of-fold** on a held-out real-image split
(2-fold cross-fit by image). It corrects the judge's scale/central-tendency bias, touches no weights,
is monotonic (rank-preserving), and — validated in §8.3 — **transfers across datasets**, so it is fit
once on real images and reused unchanged for C3.

### 5.5 The audience-guided editing loop for C4 (`src/editor/`, `script/c4_refine.py`)

A **critic-quality ablation inside a 10-step auto-refinement editing loop**, demographic-free. The
claim: a *society of personas* as the editing critic gives a better feedback signal than a single
*blind VLM* critic or a fixed *"improve this image"* string. Design principle: **critic ≠ objective**
— the audience proposes edits; an independent, held-out model judges whether the image improved.

**Loop (per image, R = 10 steps):**
1. the critic views the current best image → complaints aggregated across the panel and accumulated
   across steps into one imperative instruction (≤15 words);
2. **anchored re-edit** — the accumulated instruction is applied to the *original* image (never the
   previous output, so artifacts don't compound) by **FLUX.1-Kontext-dev**, producing K = 3
   candidates;
3. each candidate is scored by a **held-out LAION aesthetic predictor** (CLIP ViT-L/14 + MLP head — a
   different model family from the Qwen critic) and a **DINOv2 identity/drift** check;
4. **accept-if-better** — commit a candidate only if the objective improves *and* identity stays
   above the drift cap; else keep the current best. Best-so-far is monotone by construction.

**Conditions:** `static` (fixed string) · `blind` (one generic no-persona critique) · `society`
(panel of 10 PARA personas, aggregated — the method) · `reward_only` (oracle that maximizes the
objective directly — a labelled ceiling, not a fair baseline).

**Tuned defaults (to elicit visible edits):** every editor prompt gets a compact **visible-edit
emphasis** suffix (*"Make this a clearly visible edit, not a subtle one, while keeping the same
subject and scene"*), **guidance_scale 3.0**, and a **drift cap 0.78** (loosened from 0.85 so bolder
edits still commit while identity is guarded). The emphasis is constant across conditions, so it does
not bias the society-vs-blind comparison.

*Why not EditReward-Bench as the objective:* it scores instruction-*adherence* relative to a given
instruction, so it would reward whichever condition writes the easiest-to-follow instruction — a
confound. The held-out aesthetic predictor is instruction-free and independent of the proposer.

---

## 6. Experimental design and metrics

All metrics live in `src/metrics/` (and `scripts/analysis/`); every headline number carries a
**1000× bootstrap 95% CI**, clustered by rater and by image where both apply.

- **Exp 0 (ceiling).** Inter-rater reliability ICC(1)/ICC(k) and mean pairwise Spearman per image;
  variance decomposition (image / user / persona-explained ΔR² / residual); predictability ceiling
  ≈ 1 − noise-fraction.
- **C1 (group prediction).** Wasserstein-1, KL, ECE between predicted and observed *group*
  distributions, pooled per slice; and the decisive metric — **between-group separation** =
  `corr(predicted slice-to-slice gaps, observed gaps)`, image-controlled, vs a no-persona control
  (≈0 expected). Slices: age × art-familiarity × (nationality where available).
- **C2 (why aggregation works).** Aggregate-vs-individual error gap (same model); the **N-personas
  fidelity curve** (group error vs panel size N, expecting monotone decrease then saturation);
  warm-start collaborative filtering as the individual lower bound.
- **C3 (generated images).** Pairwise: aggregate the panel's per-pair preference and correlate with
  the human crowd win-rate (**C3a aggregation**); **pair-controlled between-country separation**
  (C3b cross-cultural), the direct analog of C1; controls = global-preference baseline + no-persona.
- **C4 (editing).** Best-so-far objective trajectory + **trajectory-AUC**; per-condition **final
  gain** with CI; **win-rate** (society vs blind/static); **drift-vs-step** guardrail;
  **complaint-diversity** (distinct complaints/step); qualitative before/after grid.
- **Steerability gate.** `corr(persona-induced Δ, data-empirical Δ)` + sign-agreement; run before
  trusting any persona-dependent number.
- **Leakage.** Memorization probe; seen-vs-artist-disjoint gap; fame-vs-error correlation.

---

## 7. Statistical analysis plan (pre-registered)

**Primary endpoints, fixed before test access:** (1) C1 group-distribution Wasserstein vs the
no-persona aggregate; (2) C1 between-group separation; (3) the C2 saturating-N and
aggregate-vs-individual gap; (4) the C3 aggregation correlation / cross-cultural ΔAUC; (5) the C4
society-vs-blind trajectory-AUC.

- **CIs:** 1000-sample bootstrap, clustered by rater and image.
- **Multiple comparisons:** Holm correction across the slice sweep.
- **Decision rule:** a result "counts" only if its CI excludes the null **and** it sits within the
  Exp-0 ceiling. Effect size + CI are reported, not just p-values.

---

## 8. Results

*All numbers cross-checked against `results/*.json`. Model = frozen Qwen2-VL-7B unless noted. Coverage
caveat: PARA is a 4,000-image sample, LAPIS ~34%, EVA complete.*

### 8.1 Exp 0 — the ceiling (premise holds cleanly) ✅

One-way random-effects decomposition on the normalized [0,1] scale:

| Dataset | ICC(1) — one rating | ICC(k) — group mean | between-image variance | persona-ΔR² (within-image) |
|---|---|---|---|---|
| PARA | 0.470 | **0.958** | 0.470 | 2.0% |
| EVA | 0.223 | **0.906** | 0.223 | 0.9% |
| LAPIS | 0.188 | **0.839** | 0.188 | 4.0% |

An individual rating is mostly idiosyncratic (only 19–47% shared, predictable variance); the group
mean is highly reliable (ICC(k) 0.84–0.96). Persona attributes explain only 1–4% of *individual*
taste — low exactly as predicted, and the quantitative motivation for aggregating. A classical
collaborative-filtering model confirms this from the non-VLM side (personalization gain only
+0.01–0.03 MAE).

### 8.2 Steerability — the persona is functional ✅

Does the persona move the judge in the direction the data says it should?

| Dataset | score-level r | rationale-text AUC (persona vs blind) |
|---|---|---|
| PARA | +0.367 | 0.578 vs 0.498 |
| LAPIS | +0.394 | 0.695 vs 0.506 |
| EVA | −0.239 | **0.701** vs 0.511 |

On attribute-rich datasets the persona is weakly-positive at the score level. Decisively, the persona
**steers the rationale language** far above chance on *all three* datasets (a rater's attribute is
recoverable from their generated text at AUC 0.58–0.70 vs ≈0.50 for the no-persona control), and
personas produce ~4× more distinct rationales. Most tellingly, **on EVA the persona signal is strong
in the text (0.70) even though the integer score hides it** — direct evidence that greedy decoding
censors a persona effect the model is actually computing.

### 8.3 Calibration and response style ✅

The judge exhibits a **central-tendency bias**: it uses only 51–58% of the human score range,
concentrates 34–56% of answers on one value, and avoids endpoints. This — not random error — is why
raw distributional error is large. Out-of-fold isotonic calibration corrects it:

| Dataset | group MAE (raw → calibrated) | population-mean prior | calibrated beats prior? |
|---|---|---|---|
| PARA | 0.184 → **0.062** | 0.102 | ✅ |
| EVA | 0.103 → **0.063** | 0.081 | ✅ |
| LAPIS | 0.145 → **0.072** | 0.105 | ✅ |

Calibration roughly halves group MAE and lifts the aggregate above the population prior on all three
datasets. It **transfers across datasets** (every off-diagonal fit-on-A/eval-on-B ≪ raw), which
green-lights the "fit once on real images, reuse on generated" step in C3.

### 8.4 C1 — between-group separation (the headline) 🟢

`corr(predicted slice-to-slice gap, observed gap)`, image-controlled, calibrated scores, vs a
no-persona control (≈0 expected):

| Dataset | slice | separation (95% CI) | blind | verdict |
|---|---|---|---|---|
| **LAPIS** | nationality | +0.068 [0.034, 0.104] | −0.00 | ✅ |
| **LAPIS** | art interest | +0.088 [0.067, 0.107] | +0.03 | ✅ |
| **LAPIS** | age | +0.102 [0.078, 0.127] | +0.01 | ✅ |
| **LAPIS** | **pooled** | **+0.166 [0.150, 0.181]** | +0.02 | ✅ **strong** |
| PARA | pooled | +0.044 [0.024, 0.062] | −0.01 | ✅ weak |
| EVA | pooled | −0.084 [−0.103, −0.065] | +0.01 | ✗ fails |

On LAPIS the model reproduces genuine between-group divergence (nationality, art interest, age) with
a pooled separation of **+0.17, CI excluding 0, against a no-persona control at ≈0** — the paper's
central result, and the nationality term is exactly the cross-cultural signal C3 builds on. The
effect is weak-but-real on PARA and fails on EVA. The pattern is consistent everywhere: **persona
signal scales with persona richness (LAPIS > PARA > EVA)**. It survives **Holm** correction (6/8
slice separations) and is **temperature-robust** (§8.9), so it is not a decoding artifact.

### 8.5 C2 — why aggregation works 🟡

- **Aggregate-vs-individual gap** (calibrated): group MAE beats the population prior on **all 16
  rated axes** across the three datasets; individual error ≫ group error, consistent with Exp 0.
- **Structured signals:** PARA's social axes carry a positive persona effect (light/aesthetic/
  content-preference/willingness, within-image value +0.02–0.03).
- **The aggregation *mechanism* (N-curve) is not shown on real images.** Under greedy decoding the
  personas collapse to near-identical scores, so the N-personas fidelity curve is flat — the group
  win is image-quality + calibration, not persona-averaging. This is the project's honest weak point
  on real images; the mechanism *does* appear on generated images (§8.6). See §8.9 for the decoding
  diagnosis.

### 8.6 C3 — generalization to AI-generated images 🟢 / 🔴

Run `rapidata_persona`: 23,445 votes, 898 dalle-3-vs-flux pairs, 120 countries; the frozen judge,
role-playing each voter's country, picks A or B.

**C3a — the aggregation mechanism transfers ✅ (headline positive).**

| metric | value |
|---|---|
| aggregate corr (panel vs crowd, 898 pairs) | **r = +0.466 [+0.415, +0.515]** |
| individual (1 vote) accuracy | 0.532 (≈ chance) |
| panel-aggregate accuracy | **0.659** |
| always-flux majority prior | 0.600 |
| aggregate − majority lift | **+0.056 [+0.008, +0.105]** (CI excludes 0) |
| N-curve (N=1→20) | 0.588 → **0.657** (monotone) |

Aggregating the panel recovers the crowd's preference on generated images and beats the majority
prior — the **C2 mechanism, and the project's first clean rising N-curve** (real-image runs were flat
from decoding collapse; here temp-0.7 + genuinely diverse country personas + a pairwise task give
real panel variance). This validates the aggregate as a real preference signal on generated content.

**C3b — cross-cultural differentiation does not transfer 🔴 (honest negative).** Pair-controlled
between-country separation = **+0.001 [−0.017, +0.020]** (null; contrast LAPIS +0.17). The persona
panel captures the *crowd-level* preference but not *between-country* differences — consistent with
C2's logic (aggregation cancels noise and recovers the shared signal even where the persona doesn't
encode real group structure). Caveats: ≈1 human vote per (pair, country) attenuates any effect;
dalle-3-vs-flux only; no `blind` run delivered. A powered test needs the appeal-scoring design and a
no-persona run (`docs/claim3_cross_cultural.md`).

### 8.7 C4 — audience-guided editing loop 🟢 / 🟡

Run `c4_run2`: 100 images (50 PARA + 50 EVA), 4 conditions, 10 steps, FLUX.1-Kontext, held-out LAION
aesthetic + DINOv2 drift.

**Premise validated ✅.** A frozen synthetic-audience critic drives a non-circular, identity-
preserving editing loop that measurably improves a **held-out** objective:

| condition | held-out aesthetic gain (95% CI) | identity (final) | % improved |
|---|---|---|---|
| static | +0.164 [+0.135, +0.196] | 0.968 | 84% |
| blind | +0.155 [+0.126, +0.185] | 0.964 | 72% |
| **society** | **+0.170 [+0.136, +0.207]** | 0.951 | 69% |
| reward-only (oracle) | +0.187 [+0.157, +0.221] | 0.960 | 86% |

Every condition's gain CI **excludes 0** (start 5.10 → society end 5.27), committed edits stay far
above the 0.78 identity cap, and the **society critic produces 3.1× richer feedback** (3.12 vs 1.00
distinct complaints/step). The reward-only oracle behaves as a clean ceiling; convergence is fast
(median step 3–4). This is the non-circular validation the teammate's self-graded loop could not
provide.

**Critic-ablation null 🟡 (scoped honestly).** On this *generic* aesthetic objective the three critic
conditions are statistically indistinguishable (society − blind = +0.015, CI [−0.010, +0.044]).
Likely a **measurement ceiling**: a universal aesthetic rewards generically-prettier edits that any
instruction achieves — even the oracle only reaches +0.187 — so it cannot resolve *audience-specific*
critique quality. Isolating a society-over-blind advantage requires a **group-preference-aware
evaluator** (score each edit by the target panel's predicted preference, held out from the proposer)
or a human study — the natural next experiment, which this validated loop is ready to run.

### 8.8 Bias / fairness ⚠️

Per-subgroup calibration error: the judge is **fair** across personality, age, gender, and expertise
(gaps ≤ 0.04) but shows a **large systematic bias by nationality (gap 0.50) and region (0.23)**. This
is an ethics-appendix headline and a C3 confound — hence C3's global-preference and no-persona
controls, which net it out.

### 8.9 Decoding limitation (the temperature study) — a characterized negative

The flat N-curve (§8.5) is a **decoding artifact, not evidence against aggregation.** Under greedy
decoding, ~48–51% of PARA/EVA images produce a point-mass panel. A temperature-0.7 re-run did **not**
fix it: Qwen2-VL-7B is so peaked on discrete rating tasks that it reverts to the mode ~88% of the
time, so the panel gains no spread and the N-curve stays flat. **Silver lining:** every C1 separation
is *unchanged* between temp-0 and temp-0.7 (LAPIS +0.17 → +0.17), so the headline is not a decoding
artifact. The principled fix (parked) is to read the **token-level score distribution** (softmax over
valid score tokens) in one forward pass rather than sampling — documented in
`docs/task_temperature_rerun.md`.

### 8.10 Leakage — clean ✅

No memorization: artist fame (rating volume) vs model error is +0.03 ≈ 0; the model is merely weaker
on abstract/minimalist styles (difficulty, not leakage). Parse rate ≈ 100%; the grid-snapping step is
a no-op. The zero-shot pipeline uses no AVA exemplars, so EVA⊂AVA opens no leakage path.

---

## 9. Ablations (all frozen — the method's only knobs)

1. **Persona-card fields** — drop each field group, report ΔWasserstein.
2. **Zero-shot vs few-shot** (retrieved exemplars, sweep k).
3. **Self-consistency** (T samples vs one).
4. **Calibration** — with vs. without (done: §8.3, a prerequisite not an option).
5. **Aggregation vs single judge**, and the persona-sampling scheme — core to C1/C2.
6. **Backbone** — the pinned VLM vs. additional frozen VLMs (robustness, not correctness).
7. **C4 objective source** — aesthetic vs. a group-preference evaluator (the key follow-up, §8.7).
8. **C4 editor** — FLUX.1-Kontext vs. Qwen-Image-Edit (the teammate's self-refine baseline).

---

## 10. Limitations and honest negatives

These are reported in full because each sharpens the contribution:

1. **N-curve flat on real images (§8.5, §8.9)** — greedy decoding collapses the panel; the aggregation
   *mechanism* is shown on generated images (§8.6) but not yet on real ones. Fix: token-level score
   distribution.
2. **C3 cross-cultural null (§8.6b)** — under-powered (≈1 vote/cell) and dalle-3-vs-flux only; needs
   the appeal-scoring re-run + a blind control.
3. **C4 critic-ablation null (§8.7)** — a generic aesthetic can't resolve audience-specific critique;
   needs a group-preference evaluator.
4. **EVA fails persona claims** — thin personas; consistent with the richness ordering.
5. **National/regional calibration bias (§8.8)** — flagged and controlled.
6. **Coverage** — PARA 4k-image sample, LAPIS ~34%.

---

## 11. Roadmap

**Highest value (each closes one negative above):**
1. **Token-level score elicitation** — unblocks the real N-curve (C2 mechanism on real images) and
   the distributional half of C1.
2. **C3 appeal-scoring re-run** (temp 0, `rapidata_full` + `rapidata_blind`) — a powered
   cross-cultural test with the no-persona control.
3. **C4 group-preference evaluator** — score edits by the target panel's predicted preference (held
   out from the proposer) to give society a fair shot at beating blind; optional human study.

**Stretch:** additional frozen backbones; a second editor; the optional human study (Appendix G).

---

## 12. Reproducibility, code, and artifacts

- **Judge / audience:** `src/persona/` (frozen VLM backends), `src/data/persona.py`, `src/audience/`,
  `src/predict/` (inference + calibration).
- **C4 editing loop:** `src/editor/` (`flux_editor`, `objective`, `drift`, `critic`, `loop`),
  `script/c4_refine.py`.
- **Dataset pipelines:** `script/{para,eva,lapis,rapidata}_pipeline.py`.
- **Analysis suite (28+ scripts, no GPU):** `scripts/analysis/*.py` — `exp0_ceiling`, `c1_separation`,
  `c2_ncurve`, `calibration`, `steerability`, `bias`, `leakage`, `c3_rapidata`, `c4_trajectory`,
  `c4_qualitative`, `c4_selfrefine`, … → `results/*.json` + `results/figs/*.png`.
  Shared plot theme in `scripts/analysis/theme.py`; publication figures for the paper in
  `scripts/analysis/paper_extra.py` (see Appendix F.1). All plotting scripts are seeded
  (`default_rng(0)`), so figures regenerate deterministically without changing any reported number.
- **GPU setup / run:** `scripts/setup_c4.sh`, `scripts/run_c4.sh`, `requirements-gpu.txt`,
  `notebook/c4_colab.ipynb`; H100 runbook `docs/RUNNING_ON_H100.md`.
- **Every run logged** with seed, config hash, model id, prompt version, dataset version; sharded
  chunked JSON (`<run>.shard*.part-*.json`). `data/` + `results/` git-ignored, synced to private HF.
- **Result artifacts on disk now:** `data/results/{para,eva,lapis}_{full,blind,full_t07}`,
  `rapidata_persona`, `c4_run2` (100-image C4), `c4_selfrefine` (teammate baseline).

---

## 13. Ethics

Outputs are framed as **dataset-sampled group distributions, not national or group essences.** We
report and net out the model's own national/regional rating bias (§8.8). Releases respect dataset
licenses (PARA/LAPIS redistribution-restricted → derived artifacts stay private; EVA is CC0; Rapidata
public). Human-study protocols (Appendix G) carry consent and de-identification.

---

## Appendix A — Judge prompt and output schema

**System (persona):** *"You are role-playing as one specific human annotator … Judge it the way this
exact person would … Do not mention that you are an AI or that you are role-playing."* No-persona
control drops the "as this person" framing.

**User:** `[IMAGE]` + persona card + schema:
```json
{"score": 0.0, "emotion": "awe|amusement|…|other", "willingness_share": 0.0,
 "difficulty": 0.0, "rationale": "<=25 words"}
```
Decoding: greedy (temp 0) default; JSON/schema mode when supported, tolerant parse + one retry then
NaN. **C4 critic prompt:** score + one imperative edit instruction ≤15 words + short comment.

## Appendix B — Baselines

| Baseline | Tests | How |
|---|---|---|
| No-persona aggregate (C1) | does the persona matter? | empty profile, aggregate N copies |
| Population mean (C1) | the group floor | per-image mean of reference raters |
| Single judge (C2) | value of aggregation | N = 1 point on the curve |
| Features+metadata regressor (C2) | non-VLM personalization | frozen CLIP/DINOv2 ⊕ one-hot persona → GBM/MLP |
| Collaborative filtering (C2) | warm-start individual reference | matrix factorization, warm-start only |
| Global preference (C3) | universal quality | mean win-rate ignoring slice |
| No-persona prompt (C3) | persona effect | empty-profile judge |
| Static / blind / reward-only (C4) | critic quality | fixed string / one generic critique / objective-max oracle |

## Appendix C — Metric definitions

Spearman ρ (`scipy`), MAE on `score_norm`, Wasserstein-1 per slice, KL over 10 shared bins (+ε),
ECE (10 bins), ICC(2,k) (`pingouin`/variance components), between-group separation =
`pearsonr(pred gaps, obs gaps)`, C3 aggregate corr, C4 trajectory-AUC (trapezoid of best-so-far),
bootstrap CI (1000×, clustered by rater and image).

## Appendix D — Example config (`configs/predict_persona.yaml`)

```yaml
model: Qwen/Qwen2-VL-7B-Instruct        # frozen, local
prompt: {mode: zero_shot, self_consistency: {samples: 1, temperature: 0.0}}
calibration: {method: isotonic, fit_on: splits/calib_real.json}
eval: {regime: cold_start, slices: [age_bucket, art_exp, nationality], bootstrap: 1000}
```

## Appendix E — Cost / compute

Everything is inference (no training). Exp 0 is CPU-only. Judge inference caches by
`(image, persona, model, prompt)` so panels/slices/N-sweeps reuse one query per pair. C4 full run
≈ 100 img × 4 cond × 10 steps × 3 candidates ≈ 12k FLUX edits (~hours on one H100; ÷N across shards).

## Appendix F — Deliverables → claim → artifact

| Output | Claim | Artifact |
|---|---|---|
| Variance/ceiling | Exp 0 | Fig: variance bars |
| Group separation + distribution error | C1 | Table + per-slice reliability |
| Aggregate-vs-individual gap + N-curve | C2 | Fig: N-curve |
| Aggregation transfer + cross-country | C3 | `c3_*.png`, `c3.json` |
| Editing trajectory + gain + drift + diversity | C4 | `c4_*.png`, `c4.json`, `c4_summary.md` |
| Subgroup calibration | ethics | Appendix |

## Appendix F.1 — Paper and publication-grade figures (added 2026-07-29)

The NeurIPS 2026 **Creative AI Track** submission lives in `docs/paper/neurips_creative_ai/`
(`autopolish.tex` + `figs/`; one self-contained directory per venue, since page limits and
templates differ enough that figures are re-tuned per venue;
system named **AutoPolish**). All figures share one colorblind-validated palette
(blue `#2a78d6` = ours/audience, orange `#eb6834` = control/secondary, aqua `#1baf7a`
= oracle/edit, gray = neutral; per-dataset map PARA→orange, EVA→blue, LAPIS→green),
installed via `scripts/analysis/theme.py` and applied by every plotting script.
In-figure titles were removed (message goes in the caption, NeurIPS convention).

**Main-text figures** — the workshop caps the main paper at 6 pages excluding references, so
the main text now carries three composite figures built by `scripts/analysis/paper_figs.py`
(run as `python paper_figs.py --c4-root ../../data/results/c4_run2`), plus the TikZ system
diagram (Fig. 1):

| Output | Panels | Replaces (now in the supplement) |
|---|---|---|
| `figs/pf_audience.png` | (a) calibration, (b) C1 between-group separation, (c) C3 panel-size curve | `b4_calibration.png`, `c1_separation.png`, `c3_ncurve.png` |
| `figs/pf_qualitative.png` | tight 3-row source/edit grid (top-3 society gains, `c4_run2`) | first 3 rows of `c4_qualitative.png` |
| `figs/pf_autopolish.png` | (a) best-so-far trajectory, (b) gain vs DINOv2 identity | `c4_trajectory.png`, `c4_headline.png` (this one lives in the supplement) |

These are drawn at their **final printed width (5.5in) with 6-7pt type** and included at
`width=\linewidth`; `paper_figs.print_size()` sets the rcParams. Drawing wide and letting
LaTeX scale down (the earlier approach) rendered axis and cell labels at ~3pt in print.

Displaced to the supplement to fit 6 pages, with every claim kept in the main text:
the dataset table, the ceiling table (its per-dataset ICCs are now inline in Sec. 5.1),
`b1_steerability.png`, `c3_aggregate_scatter.png`, `c4_headline.png`, the full 5-row
`c4_qualitative.png`, and `pf_autopolish.png`. The main text makes **no reference to the
supplement** — it is self-contained.

**Appendix figures/tables** (new, from `scripts/analysis/paper_extra.py` + cached JSONs):

| Output | Backs | Artifact / source |
|---|---|---|
| Calibration transfer heatmap (fit-on × eval-on) | calibration transfers → "fit once, reuse on C3" | `figs/ax_calib_transfer.png` ← `calib_transfer.json` |
| Central-tendency bias (modal share + entropy, human vs VLM) | why calibration is needed | `figs/ax_response_style.png` ← `response_style.json` |
| Group beats prior on all 16 axes (scatter vs diagonal) | breadth of C2 aggregate win | `figs/ax_breadth.png` ← `dims_extended.json` (+ `cf_baseline.json`) |
| Subgroup fairness severity bars | ethics: fair except nationality/region | `figs/ax_bias.png` ← `bias.json` |
| Temperature robustness table (t0 vs t0.7) | headline not a decoding artifact | `temp_compare.json` |
| Holm-corrected slice tests table (6/8 survive) | C1 multiple-comparison control | `holm.json` |
| Rationale steerability (AUC persona vs blind + diversity) | persona functional in the text even when the score hides it | `figs/ax_rationale.png` ← `rationale.json` |
| Content-category difficulty bars (night scenes hardest) | error is content-difficulty, not artifact | `figs/ax_content_category.png` ← `content_category.json` |
| Robustness & validity checks table (memorization, duplicate-invariance, agreement, difficulty, structure) | group signal is genuine, not leakage/bookkeeping | `leakage.json`, `repeated_measures.json`, `accuracy_vs_agreement.json`, `validity_difficulty.json`, `structure.json` |
| Second qualitative before/after grid (7 society-wins rows) | more cherry-picked strong AutoPolish edits | `figs/ax_qualitative2.png` ← `c4_qualitative.py --skip 5 --n-show 7 --society-top` on `c4_run2` |
| Refinement-over-time grid (best-so-far at steps 0/2/4/6/10, 7 rows) | shows the auto-research loop polishing each image progressively | `figs/ax_progression.png` ← `c4_progression.py` on `c4_run2` |

Supplement is a standalone section (own title block, S-numbered sections/figures/tables) after the references, and now opens with: **S1 Prompts** (persona judge, no-persona control, and complaint-to-instruction distiller, in `tcolorbox` prompt boxes, verbatim from `src/persona/person.py`, `src/editor/critic.py`, `script/para_pipeline.py`); **S2** a worked example of the panel reacting to one image (5 of 10 real panelist reactions + scores, from `c4_run2` society logs); **S3** the aggregation of those complaints into one distilled edit instruction with before/after images (office: `ex_office_{src,edit}`, basketball: `ex_ball_{src,edit}`).

Note: `c4_qualitative.py` now resolves the pre-edit source from the saved `step0_source.png` or, when absent locally, the dataset original under `data/<ds>/` (recursive search); it also takes `--skip/--n-show/--out-name/--title`. The main-text Fig.~7 keeps the original complete render (some per-image edit PNGs are not synced to every machine, so a local re-render can show blank cells).

## Appendix G — Optional human study

Prolific/in-house, consent + de-identified intake (PARA/LAPIS fields + BFI-10). Part A
(psychographic groups): ~300 generated images, ≥10 raters each, endpoint = pooled subgroup
distributional error. Part B (extends C4): forced choice over the final images of the three loop
conditions (society/blind/static), endpoint = society-vs-blind win-rate. Power for the pooled
endpoint; IRB in week 1; the paper does not depend on this.

## Appendix H — Datasets and access

`scripts/fetch_from_hf.py {para,eva,lapis}` (private, token + repo access) and Rapidata (public
pull). ISO-2 → LAPIS-nationality map needed for the C3 LAPIS-overlap subset. See `docs/{para,eva,
lapis,claim3_cross_cultural}.md` for per-dataset notes and the C3 spec.
