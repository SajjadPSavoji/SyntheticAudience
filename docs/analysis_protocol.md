# Analysis Protocol & Audit — Synthetic Audiences (first run-through)

Scope: what to do with the results of the teammate's first pass (the six `data/results/*`
folders) to (a) **audit** them before trusting any number and (b) **turn them into the
experiments the research plan actually asks for**. This maps to `PROPOSAL.md` /
`research_plan.md` claims **Exp 0**, **C1**, **C2**, **C3**, **C4**.

> One-line verdict: the runs are a solid raw-material dump for the **individual-level (C2
> lower-bound)** analysis, but **none of the headline analyses exist yet** (Exp 0 ceiling,
> steerability gate, C1 group aggregation + between-group separation, N-curve, calibration,
> bootstrap CIs), and there are **three confounds** (no calibration, a temperature mismatch
> between the two modes, and a model that diverges from the plan) that must be handled before
> any number is quotable.

---

## 0. What was actually run

| Run (`data/results/`) | Dataset | Mode | Model | Temp | Ratings | Images | Dims |
|---|---|---|---|---|---|---|---|
| `para_full`  | PARA  | persona-conditioned | Qwen2-VL-7B-Instruct (local) | 0.0 | ~51.7k | 2000 | 9 |
| `para_blind` | PARA  | persona-blind (generic prompt) | " | 0.7 | ~51.7k | 2000 | 9 |
| `eva_full`   | EVA   | persona-conditioned | " | 0.0 | ~137k | 4070 | 6 |
| `eva_blind`  | EVA   | persona-blind | " | 0.7 | ~137k | 4070 | 6 |
| `lapis_full` | LAPIS | persona-conditioned | " | 0.0 | ~91k | 4000 | 1 |
| `lapis_blind`| LAPIS | persona-blind | " | 0.7 | ~91k | 4000 | 1 |

- **full** = the plan's persona-conditioned judge: each real rater is re-created as a persona
  card (stored per-run in `users`) and the VLM answers *in character*.
- **blind** = the plan's **no-persona control** (`*_GENERIC_SYSTEM_PROMPT`): one rater-agnostic
  prompt, run on the *same* (image, rater) tasks and scored against each rater's true value.
- Each record is one `(image, user)` row with `gt_*` (the human's real ratings), `pred_*`
  (the VLM's), plus `comment`/`raw_response`. Persona cards live in `users` as free text (the
  structured attributes — age, gender, nationality, Big-5, art/photo experience — are only
  *inside* the card string; to slice we must re-join `userId` → attributes from the source
  annotation CSVs in `data/{para,lapis,eva}`).
- Already computed per run: `metrics.per_dimension[dim]` with **per_rating** (MAE, RMSE,
  mean_bias, pearson, spearman, and two baselines: `baseline_mae_global_mean`,
  `baseline_mae_image_mean`) and **per_image** (n, human_mean/std, vlm_mean/std, MAE, EMD, KS,
  and `human_resample_emd` = a bootstrap noise floor).

**Deviation from the plan to record up front:** the plan mandates a *frozen serverless HF
Inference-Providers VLM* (GLM-4.5V / aya-vision / …). These runs used a **local Qwen2-VL-7B**.
That is fine as a first pass but the "serverless-only" headline still has to be reproduced on a
provider model, and Qwen2-VL-7B becomes a "rejected/secondary" backbone unless it is
provider-callable.

---

## 1. Preliminary signal (pooled across all 4 shards, **uncalibrated**) — read with caution

Primary dimension only (PARA `aestheticScore` 1–5, EVA `score` 0–10, LAPIS `rating` 0–100):

| run | MAE | mean_bias | Spearman | baseline MAE (global mean) | baseline MAE (image mean) |
|---|---|---|---|---|---|
| para_full  | 0.828 | **+0.648** | 0.493 | 0.611 | **0.438** |
| para_blind | 0.639 | +0.276 | 0.533 | 0.611 | 0.438 |
| eva_full   | 1.669 | +0.737 | 0.277 | 1.630 | 1.423 |
| eva_blind  | 1.761 | +1.030 | 0.315 | 1.630 | 1.423 |
| lapis_full | 23.34 | +13.11 | 0.339 | 22.98 | 19.56 |
| lapis_blind| 24.71 | +16.30 | 0.290 | 22.98 | 19.56 |

Four things jump out — each becomes an audit/analysis item below:

1. **No run beats the "predict the image's own mean" baseline on MAE.** This is **expected and
   is the entire C2 thesis**, not a failure: individual taste sits near the noise ceiling, so an
   oracle that already knows each image's true mean is hard to beat *at the individual level*.
   The image-mean baseline is essentially the **C1 group target**; the real question (unanswered
   so far) is whether the *aggregated* prediction reaches it and captures *between-group*
   structure. → drives **B0, B2, B3**.
2. **Personas don't obviously help.** On PARA and EVA the **blind** run matches or beats **full**
   on rank correlation (0.533 vs 0.493; 0.315 vs 0.277); only LAPIS shows persona > blind
   (0.339 vs 0.290). This puts the **steerability gate at risk** and must be measured directly,
   not eyeballed. → drives **B1**.
3. **Large positive bias everywhere** (VLM systematically over-rates: +0.65 on a 1–5 scale,
   +13 on 0–100). Any raw MAE/EMD comparison is therefore unfair to the VLM and unstable.
   **Calibration is mandatory before any distributional claim.** → drives **B4**.
4. **The two modes are not a clean comparison.** `full` ran at **temp 0.0** (greedy) and
   `blind` at **temp 0.7**. Fraction of images whose predicted distribution is *degenerate*
   (`vlm_std == 0`): para_full 48% / para_blind 88%; eva_full 51% / eva_blind 90%; lapis_full
   4% / lapis_blind 86%. So full-vs-blind mixes *persona conditioning* with *decoding
   temperature*, and every per-image EMD/KS for the `full` runs is "distance from a near-point
   mass to a human distribution." → drives **A5, A7, B2**.

---

## Part A — AUDIT (do this before quoting a single number)

Each item: **what / why / how**. Everything in Part A is pure re-analysis of the existing logs
(no GPU, no inference).

**A1. Recompute every metric from the raw `(gt_*, pred_*)` records; do not trust the baked-in
`metrics` block.** Why: we need to know the numbers are reproducible and to control binning,
weighting, and NaN handling ourselves. How: load all `*.part-*.json`, rebuild per_rating and
per_image, and assert they match `metrics.per_dimension` to tolerance. Flag any mismatch.

**A2. Parsing integrity.** Why: small VLMs emit malformed JSON; how unparsed rows are counted
silently changes every average. How: (a) compute the **parse-failure rate** per run
(`pred_* is None`/NaN) — is it uniform across images, or concentrated (a biased dropout)? (b)
confirm `pred_*` actually derives from `raw_response` (spot-check re-parse); (c) check
**clamping / out-of-range** (any pred outside the dimension's [min,max]); (d) confirm dropped
rows are excluded from *both* the VLM error and the baselines, not just one.

**A3. Ground-truth & scale integrity.** Why: a single scale slip poisons MAE and calibration.
How: re-join a sample of `(userId, imageName) → gt_*` against the source CSVs in
`data/{para,lapis,eva}`; confirm normalization matches the plan (PARA 1–5, EVA 0–10, LAPIS
0–100 as stored here — note the plan's unified schema wants `score_norm∈[0,1]`, so decide one
canonical scale and convert *once*). Confirm `pred` and `gt` share the same scale per dimension.

**A4. Coverage & pairing.** Why: full-vs-blind and any aggregation must be over identical tasks.
How: confirm `full` and `blind` cover the **same (image, user) set** per dataset (set-diff
should be empty); report the **raters-per-image** distribution (the plan assumes ~25–35);
confirm the LAPIS Latin-1-mangled/missing images were dropped consistently; run the **EVA↔AVA
dedup leakage check** the plan calls for (EVA images are an AVA subset).

**A5. Decoding confound (blocking for full-vs-blind).** Why: see red flag #4. How: document the
temp mismatch explicitly; then either (i) restrict all full-vs-blind contrasts to
**decoding-invariant** quantities (rank correlation, per-*image-mean* error) rather than
distributional EMD/KS, or (ii) schedule a **matched-temperature re-run** (Tier 2) so the two
modes differ *only* in persona conditioning.

**A6. Persona-blind sanity.** Why: the control only works if it truly ignores persona. How:
confirm the blind system prompt is the generic `*_GENERIC_SYSTEM_PROMPT`, that no persona text
leaked in, and that blind still ran the full (image, user) task list (so it is scored against
the same per-rater truths).

**A7. Degenerate-distribution accounting.** Why: with `vlm_std==0`, EMD collapses to
`|vlm_mean − human_mean|` and KS is trivially large; comparing that to `human_resample_emd`
(the noise floor) is apples-to-oranges. How: report the degenerate fraction per run (done
above), and mark every distributional metric on `full` (temp 0) as **not a distribution match**
until a temperature-bearing run exists.

**A8. Baseline honesty.** Why: `baseline_mae_image_mean` uses the test image's *own* raters'
mean — an **oracle** for the group, not a fair predictor. How: keep it, but relabel it as the
**empirical group ceiling / C1 target**, not a "baseline the model should beat" at the
individual level. `baseline_mae_global_mean` (one constant for all) is the true individual
floor; the VLM must beat *that*.

**A9. Manifest completeness & duplicates.** Why: sharded exports can silently drop chunks. How:
assert `sum(len(part)) == n_ratings` per shard and across shards; check for duplicate
`(image, user)` rows; confirm all `result_parts` referenced by each summary exist on disk.

**A10. Model/plan-divergence log.** Record Qwen2-VL-7B-local + temps + seed=0 as the run
configuration, and note the serverless-model reproduction as an open requirement (Tier 2).

**Audit exit criterion:** a short `results/audit.json` + `docs/audit_findings.md` stating, per
run, the parse rate, coverage, reproduced headline metrics, and which comparisons are
confound-free.

---

## Part B — ANALYSIS PROTOCOL (mapped to the plan's claims)

Status legend: **[now]** = pure re-analysis of existing logs · **[join]** = needs the
`userId → structured attributes` re-join from source CSVs · **[run]** = needs new inference.

### B0. Exp 0 — ceiling analysis **[now]** — *do this first; it reframes everything*
The `per_image` blocks already carry `human_std` and `human_resample_emd` (a bootstrap noise
floor) per image — most of Exp 0 is computable without touching the VLM.
- **Inter-rater reliability**: ICC(2,k) and mean pairwise Spearman per image, per dataset
  (`pingouin` / `statsmodels`) from the `gt_*` rows.
- **Variance decomposition**: `gt ~ 1 + (1|image) + (1|user) + persona_fixed_effects` (mixed
  model) → variance fractions (image / user / persona-ΔR² / residual). Needs **[join]** for the
  persona fixed effects.
- **Ceiling**: max individual R² ≈ `1 − noise_fraction`; persona-attributable ceiling = ΔR².
- **Gate (from the plan):** a low persona ΔR² is *expected*; only a vanishing **group** signal
  (B2 failing) triggers the "frozen VLMs can't simulate group taste" pivot.
- Output: `results/exp0.json` + variance bar chart. Every individual-level number below is
  reported *against* this ceiling.

### B1. Steerability gate **[join]** — *run before trusting any persona claim; red flag #2 says it's at risk*
- From the source data, compute each attribute's **empirical** effect on the mean score
  (e.g., art-familiarity → Δ on abstract art in LAPIS).
- From the `full` logs, compute the VLM's **predicted** shift per attribute (this is already an
  ablation-by-nature: `full` vs `blind` is the persona-on/off contrast; within `full`, regress
  `pred` on parsed persona attributes).
- **Metric:** `corr(predicted Δ, empirical Δ)` + fraction of attributes with the correct sign.
- **Gate:** if steerability ≈ 0 the frozen model ignores the persona → try a stronger
  prompt / few-shot / a bigger or serverless backbone; if still null, the paper pivots to the
  **ceiling / "frozen VLMs can't personalize"** finding (still publishable, per the plan).

### B2. C1 — group prediction (**headline**) **[now]** + **[join]** for slices — *not started*
This is the paper's headline and **does not exist in the current results**. The `full` run is
already a **panel of the real raters as personas per image**, so the aggregate is constructible:
- **Group distribution match**: aggregate the per-persona `pred` for each image (and, better,
  per **slice**) into a predicted group distribution; compare to the observed group with
  **Wasserstein-1, KL, ECE**, *pooled per slice* (never per-image-within-slice — too noisy).
  Baselines: **no-persona aggregate** (the `blind` run, aggregated) and the **population-mean
  prior**. Pass = lower distributional error than both, CI excluding 0.
- **Between-group separation (the decisive sub-result)**: slice by age × art-familiarity ×
  (nationality for LAPIS), compute observed slice-to-slice score gaps, and correlate with the
  VLM's predicted gaps. This is where a real cross-group signal lives, and it needs **[join]**.
- **Caveat:** because `full` is temp 0, the *within-group spread* is currently persona-driven
  only; a matched-temperature run (Tier 2) is needed before the ECE/spread half is clean. The
  *mean* and *between-group-gap* halves are computable now.
- Add **bootstrap 95% CIs** (1000 draws, clustered by rater and image) to every headline number.
- Output: `results/c1.json`, Table 1 + per-slice reliability diagrams.

### B3. C2 — why aggregation works **[now]** — *strongest thing producible immediately*
Everything here is computable from the existing `full` logs:
- **Aggregate-vs-individual gap**: place the per-rater individual error (already have it:
  Spearman/MAE against the Exp-0 ceiling) next to the group-level error (aggregate per image),
  and show individual error ≫ group error, on the *same* model.
- **N-personas fidelity curve**: subsample N ∈ {1,2,5,10,20,50} personas *from the raters of
  each image* and recompute the group error; expect a monotone decrease that saturates. Report
  the saturating N. (This is pure resampling of the cached predictions — no new inference.)
- **Warm-start CF reference** (individual lower bound only): `surprise` SVD/ALS on user×image,
  warm-start split — a reference point, not a group baseline.
- Output: `results/c2.json`, Fig 3 (N-curve) + the gap row.

### B4. Post-hoc calibration **[now]** — *prerequisite for B2's distributional half; red flag #3*
- Fit an **isotonic** (or temperature) map from raw `pred` → calibrated score on a **held-out
  real-image split**; apply, then recompute all MAE/EMD/KL/ECE. Changes scores only, no weights.
- Fit **once** on real images and reuse unchanged (its transfer to generated images is itself
  part of the C3 claim later).
- Report every distributional metric **both** raw and calibrated so the bias contribution is
  visible.

### B5. C3 / C4 — **[run]**, not started — *out of scope for this analysis pass; note prerequisites*
- **C3** (cross-cultural, generated images) needs the **Rapidata** runs (pairwise, sliced by
  country/language, restricted to LAPIS nationalities, scored on disagreement pairs via ΔAUC).
  Prerequisite: B1 steerability and B2 between-group must first show a persona/culture signal on
  real images — otherwise C3 has nothing to generalize.
- **C4** (editing) needs the editor + EditReward loop (proposer ≠ reranker). C4-targeted is
  contingent on C3 showing between-group variance.
- Neither can be audited from the current logs; both are new-inference workstreams.

---

## Part C — Prioritized work items & sequencing

**Tier 1 — pure re-analysis of the existing logs (no GPU, do now).** In order:
1. **A1–A10 audit** → `docs/audit_findings.md`, `results/audit.json`.
2. **B0 Exp 0 ceiling** → `results/exp0.json` (reframes all individual numbers).
3. **B4 calibration** → calibrated re-scores cached for reuse.
4. **B3 C2 gap + N-curve** → `results/c2.json` (the most defensible immediate result).
5. **B1 steerability + B2 C1 between-group** (both need the `userId → attributes` **[join]**
   from `data/{para,lapis,eva}` — build that join table once and reuse it).
6. **Bootstrap CIs** on every headline number.

**Tier 2 — new inference (schedule after Tier 1 says the signal is real).**
7. **Matched-temperature re-run** so full-vs-blind differs only in persona conditioning (fixes
   A5/A7 and unlocks the distributional half of C1/ECE).
8. **Serverless-model reproduction** of C1 + steerability on a provider VLM (GLM-4.5V /
   aya-vision) to satisfy the plan's frozen-serverless headline; Qwen2-VL-7B becomes a
   secondary/robustness backbone.

**Tier 3 — new workstreams.**
9. **C3** on Rapidata (contingent on B1/B2 signal). 10. **C4** editing loop.

---

## Part D — What each output feeds (per `PROPOSAL.md` Appendix F)

| Output | Claim | Paper artifact | Status here |
|---|---|---|---|
| Exp 0 variance/ceiling | motivates aggregation | Fig 1 | **[now]** — B0 |
| Steerability corr | validity gate | Table 1 footnote | **[join]** — B1, *at risk* |
| C1 group error + between-group | C1 (headline) | Table 1 + Fig 2 | **[join]** — B2, *not started* |
| C2 gap + N-curve | C2 | Fig 3 + gap row | **[now]** — B3, *ready* |
| Calibration on/off | ablation | Table 4 | **[now]** — B4 |
| C3 ΔAUC | C3 | Table 2 | **[run]** — B5 |
| C4 win-rates | C4 | Table 3 | **[run]** — B5 |

---

## Open decisions for the team
1. **Canonical scale**: keep each dataset's native scale, or convert everything to
   `score_norm∈[0,1]` (plan's unified schema) before computing metrics? (Affects
   cross-dataset comparability.)
2. **Matched-temperature re-run now or later** — it is the cleanest fix for the biggest
   confound, but it costs inference. Tier-1 rank/mean analyses don't need it; the ECE/spread
   half of C1 does.
3. **Serverless swap timing** — reproduce the headline on a provider VLM before or after the
   local-Qwen analysis is fully built out?
4. **Attribute join source** — parse the stored persona-card text, or re-join `userId` to the
   original annotation CSVs (cleaner; recommended). Build one join table per dataset and reuse.
