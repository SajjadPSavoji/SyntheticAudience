# Research Plan — Synthetic Audiences for Creative AI

This is the step-by-step plan to build the system, run the experiments, and produce the results described in `PROPOSAL.md`. It maps directly to the four claims there: **C1** (predicting a group's reaction — the headline), **C2** (why aggregation makes that possible), **C3** (generalizing to AI-generated images across cultures), and **C4** (using the simulated audience as editing feedback), preceded by **Exp 0** (a ceiling analysis that motivates everything).

The whole pipeline is built on Hugging Face, and every model call — the judge, the image editor, the reward/reranker — is made **serverlessly through Hugging Face Inference Providers**. We never train, fine-tune, or self-host a model backbone for the main result. Our own code only handles data processing, caching, calibration, metrics, and the classical (non-VLM) baselines.

**Five guiding rules.**
1. **Frozen and serverless only.** All VLM/editor/reward calls use off-the-shelf provider-hosted models. The only fitting we allow is (a) post-hoc calibration, which maps scores and touches no weights, and (b) classical baselines (a regressor and collaborative filtering). We never update, download, or self-host a judge backbone for the MPR.
2. **MPR first.** The Minimum Publishable Result is Exp 0 + C1 + C2 on one pinned serverless VLM. Everything else layers on top.
3. **Persona is text; the image is the only visual input.**
4. **Pre-register** the primary statistical endpoints and the population-mean baseline before looking at any test result.
5. **The proposer is never the reranker** in C4, and every headline number carries a bootstrap confidence interval.

---

## 0. Repository and environment

### 0.1 Layout

```
syntheticaudience/
  configs/            # yaml: data, model, prompt, eval
  data/               # raw/ (downloads), processed/ (parquet), splits/
  src/
    data/             # loaders + persona-card builder + split maker
    models/           # judge wrapper, reward wrappers, editor wrapper
    predict/          # frozen inference + post-hoc calibration
    eval/             # exp0, c1, c2a, c2b, c3 runners
    metrics/          # spearman, wasserstein, kl, ece, auc, bootstrap
    audience/         # persona sampling + aggregation
  scripts/            # one-command entry points per stage
  results/            # json metrics, figures, tables
  preregistration.md  # statistical endpoints fixed before test access
```

### 0.2 Environment

```bash
uv venv && source .venv/bin/activate     # or conda
uv pip install "datasets" "scikit-learn" "scipy" "statsmodels" "pingouin" \
  "scikit-surprise" "sentence-transformers" "faiss-cpu" "pandas" "pyarrow" \
  "Pillow" "matplotlib" "huggingface_hub" "openai" "httpx" "tenacity"
huggingface-cli login          # token needs Inference Providers permission
export HF_TOKEN=...            # or rely on huggingface-cli auth
```

**Hardware.** No local GPU is needed for the core MPR. *Everything is inference* — the judge, editor, and reward models are all frozen and called through provider APIs. Our costs are API volume, image transfer, and rate limits (see Appendix E).

**Serverless details to verify in week 1** (a model being on the HF Hub is not enough): the exact model IDs callable through Inference Providers; the provider suffix/policy (`:fastest`, `:cheapest`, or an explicit provider); the image input format (`image_url` vs. base64 data URL); JSON/schema support; rate limits; the billing account; content filters; and whether the FLUX and EditReward routes are available serverlessly.

---

## 1. Data: acquisition, schema, and splits

The goal is two clean tables: one **unified per-(image, rater) parquet** for the real-image datasets, and one **per-(pair, slice) parquet** for the generated-image data. Every rating is normalized to `score_norm ∈ [0,1]`.

### 1.1 The datasets and how to get them

| Dataset | Source | Used for | Action |
|---|---|---|---|
| **PARA** | [dataset page](https://web.xidian.edu.cn/ldli/en/dataset.html) → [Google Drive zip](https://drive.google.com/file/d/1ZKNceBy5eLn2XgPd2fsEQEosKfUkMdGO/view) (password-protected) | C1, C2, Exp 0 | request in week 1; expect a CSV of (image, user, score, attributes) |
| **LAPIS** | [homepage](https://sites.google.com/view/lapisdataset/homepage) → [OSF osfstorage](https://osf.io/zw39r/files/osfstorage) (`LAPIS password protected.zip`; password via [terms form](https://docs.google.com/forms/d/e/1FAIpQLSeEaohgR50NvgDBJ2h5ynsw_NOixWDuNNydpIOujm-wY4qE6g/viewform)) | C1, C2, Exp 0, and nationality for C3 | request in week 1; **audit nationality coverage** |
| **EVA** | [GitHub](https://github.com/kang-gnak/eva-dataset) (CSV, `=`-delimited; images are a resized AVA subset; **CC0-1.0**) | C1, C2, Exp 0 — a third real-group photo cohort | clone in week 1; parse the per-vote CSV into (image, user, score 0–10, 4 attributes, difficulty); **deduplicate images against any AVA-derived exemplars to prevent leakage** |
| **ArtEmis** | `datasets.load_dataset("youssef101/artelingo")` (English subset) | warm-up critique language | filter `lang == "en"` |
| **RPCD** | project page | warm-up critique language | optional; skip if blocked |
| **Rapidata** | `load_dataset("Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3")` (+ companion sets) | C3 | parse `detailed_results` for per-vote `country` and `language` |
| **RichHF-18K** *(optional)* | HF dataset / extract via Pick-a-Pic | an extra fine-grained generated-image sanity check | not required |
| **EditReward-Data / -Bench** | `load_dataset("TIGER-Lab/EditReward-Data")`, `("TIGER-Lab/EditReward-Bench")` | C4 | preference pairs + benchmark |

**Week-1 license and availability audit (blocking).** Before building anything, confirm: (a) PARA and LAPIS redistribution terms (EVA is CC0, so it is already cleared); (b) the Rapidata license and that image bytes or provider-accessible URLs are present; (c) the LAPIS nationality list; (d) whether the dataset terms permit sending images and persona metadata to hosted inference providers; and (e) which candidate VLM/editor/reranker models are actually callable through Inference Providers.

### 1.2 The unified rating schema (`processed/ratings.parquet`)

One row per (image, rater). Fields present in some datasets but not others are left null.

```
image_id, image_path, dataset {para, lapis, eva},
user_id, score_norm[0..1],

# structured reactions (coverage differs by dataset):
emotion, content_pref, willingness_share,                       # PARA only
difficulty,                                                     # PARA + EVA
eva_attr_visual, eva_attr_composition,
eva_attr_quality, eva_attr_semantic,                            # EVA only (4 attribute votes)

# persona attributes (per user):
age_bucket, gender, education, art_exp, photo_exp,
big5_O, big5_C, big5_E, big5_A, big5_N, nationality, region
```

**Normalization.** PARA score (1–5) → `(s − 1) / 4`; LAPIS (0–100) → `s / 100`; EVA (integer 0–10) → `s / 10`.

**What EVA does and does not fill.** On the persona side EVA populates only `age_bucket`, `gender`, `region`, and `photo_exp` — it has no education, art familiarity, Big-5, or nationality. On the reaction side it fills `difficulty` and the four `eva_attr_*` votes. Because EVA records only a coarse `region` rather than `nationality`, it strengthens the C1/C2 headline but does **not** enter the C3 cross-cultural slice.

### 1.3 Splits (`splits/`) — the C1 design hinges on this

The "reference pool" is the set of ratings we are allowed to use for fitting calibration, drawing few-shot exemplars, and fitting the classical baselines. The frozen VLM never sees any of it as gradients — the term just marks what is fair game for the non-VLM machinery.

- **cold_start** — hold out a disjoint set of *users* (e.g., 20%); none of their ratings enter the reference pool. This is the **primary C1 regime**: the model predicts for a viewer it has no data on, from the persona card alone.
- **warm_start** — the same users appear in both pool and test, but we hold out specific (user, image) pairs. The collaborative-filtering baseline lives here only.
- **image-disjoint** (and, for LAPIS/WikiArt, **artist-disjoint**) test folds for leakage control.
- Save all split indices as JSON and fix a global `seed = 0`.

### 1.4 The generated-image parquet (`processed/rapidata_pairs.parquet`)

One row per (pair, slice).

```
pair_id, prompt, image_a_path, image_b_path, model_a, model_b,
slice_key (country | language), votes_a, votes_b, n_votes
```

For each (pair, slice), compute `winrate_a = votes_a / (votes_a + votes_b)`. Keep only slices with `n_votes ≥ 30`, restrict to the top-N slices by volume, and restrict to nationalities that also appear in LAPIS.

---

## 2. The persona card (the conditioning input)

The persona card is a deterministic text template built from a user's attributes. Missing fields are simply omitted — never invented.

```
src/data/persona.py → build_card(user_row) -> str
```

Example:

```
Viewer profile: 35–44 years old, master's degree, high art familiarity,
moderate photography experience, nationality: Belgium.
Personality (Big-5, 1–5): Openness 4.2, Conscientiousness 3.1,
Extraversion 2.8, Agreeableness 3.9, Neuroticism 2.5.
```

For Rapidata (C3) only `country` and `language` are available, so the card there uses only those fields. **This is critical:** the card builder must emit the *same field set* in the reference pool and at C3 test time for any field we want to claim generalization on. That is why the cross-cultural claim covers only `nationality`/`country`.

---

## 3. Models (Hugging Face Inference Providers wrappers)

The judge is a single swappable class wrapping a serverless VLM.

```python
from huggingface_hub import InferenceClient

class Judge:
    def __init__(self, model_id, provider="auto", bill_to=None):
        # model_id examples after the week-1 audit:
        # "zai-org/GLM-4.5V", "CohereLabs/aya-vision-32b:cohere",
        # or any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM.
        self.model_id = model_id
        self.provider = provider
        self.client = InferenceClient(provider=provider, bill_to=bill_to)

    def score(self, image, persona_card) -> dict:
        # user content = [image_url or base64 data URL, persona_card, output-format]
        # request response_format=json_schema when the provider supports it
        # parse JSON {score, emotion, willingness, difficulty, rationale}
```

- **Judge.** The unified default is one HF Inference Providers conversational VLM, pinned after the week-1 audit. Seed candidates: `zai-org/GLM-4.5V`, `CohereLabs/aya-vision-32b:cohere`, plus any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM. The old open-weight shortlist is part of the MPR only if those exact models become serverless-callable.
- **Serverless audit log.** For every candidate, record the model ID, provider, task support, image-input support, JSON/schema support, max context/output, pricing and rate limits, content filters, and smoke-test parse rate. Keep the rejected candidates too.
- **Output format.** Request JSON/schema mode when supported, and also force JSON in the prompt; parse with a tolerant parser; on a parse failure, retry once and then mark the value NaN.
- **Reward wrappers** (`src/models/rewards.py`). Serverless-first: use EditReward only if a provider route exists; otherwise evaluate C4 with EditReward-Bench labels and serverless scalar evaluators, always keeping proposer ≠ reranker.
- **Editor** (`src/models/editor.py`). Serverless-first: audit `black-forest-labs/FLUX.1-Kontext-dev` via a provider route, with SD-3.5-Medium (or another provider-backed editor) as fallback.

---

## 4. Exp 0 — Ceiling analysis (run FIRST)

This step bounds how predictable individual taste can even be, and in doing so it *motivates* aggregation: it quantifies how much idiosyncratic variance the group average is expected to cancel. Every individual-level number later is read against this ceiling.

`src/eval/exp0_ceiling.py`:
1. **Inter-rater reliability** — ICC(2,k) and mean pairwise Spearman across raters per image, computed for PARA, LAPIS, and EVA separately.
2. **Variance decomposition** via a linear mixed model (`statsmodels`):
   `score_norm ~ 1 + (1|image) + (1|user) + persona_fixed_effects`.
   Report the variance fractions: image, user, persona-explained (the ΔR² of the persona fixed effects), and residual noise.
3. **Predictability ceiling** — the maximum achievable individual-level R² ≈ `1 − noise_fraction`; the persona-attributable ceiling is ΔR²(persona).
4. **Output** — `results/exp0.json` plus a variance bar chart. This number is reported next to every individual-level result and anchors C2's aggregate-vs-individual gap.

**Decision gate.** A low persona ΔR² is *expected and fine* — it is precisely the motivation for aggregating. Only if the *group* signal also vanishes (i.e., C1 fails) do we pivot the paper toward "frozen VLMs cannot simulate group taste."

---

## 5. Stage 1 — The AudienceJudge (frozen, no training)

### 5.1 The prediction recipe (`src/predict/`)

The judge is a frozen serverless VLM queried with a prompt through HF Inference Providers. Its output is JSON — `{score, emotion, willingness, difficulty, rationale}` (see Appendix A). The only tunable parts are the prompt and the post-processing; the weights are never touched.

1. **Zero-shot** persona-conditioned prompting — the default per-persona judge mode. (The headline method is panel-and-aggregate over personas; see §6.)
2. **Few-shot in-context** (optional) — prepend k = 2–8 exemplars, either a fixed diverse set or nearest neighbours retrieved from the reference pool in CLIP/DINOv2 image space (and/or persona space) via `faiss`. Each exemplar carries its (persona, reaction) so the model copies the mapping in context.
3. **Self-consistency** (optional) — sample T outputs at temperature > 0 and aggregate (median score, majority emotion).
4. **Post-hoc calibration** (`src/predict/calibrate.py`) — fit an isotonic (or temperature) mapping from raw to calibrated score on a held-out real-image split. This corrects scale and bias for the distributional metrics and changes no weights. It is fit once on real images and reused unchanged for C3 (its generalization there is part of the C3 claim).

**Provider mechanics.**
- Use `AsyncInferenceClient` / `httpx` workers with bounded concurrency, exponential backoff, and explicit handling of 429/5xx and provider failover.
- Store the raw request/response metadata: model ID, provider, policy suffix, prompt version, decoding params, response format, token/image usage where returned, latency, and parse status.
- Make images provider-accessible: prefer local byte-to-data-URL only if the provider supports it; otherwise use release-safe signed or public URLs generated during preprocessing.

### 5.2 The steerability gate (`src/eval/steerability.py`) — run before trusting any number

- Build **data-driven expected directions**: from the reference pool, compute each attribute's empirical effect on the mean score (e.g., high art familiarity → +Δ on abstract art).
- Permute or ablate fields in the persona card given to the frozen judge, measure how the prediction shifts, and correlate those shifts with the empirical directions.
- **Metric:** steerability = corr(predicted Δ, empirical Δ), plus the fraction of attributes with the correct sign.
- **Gate:** if steerability ≈ 0, the frozen model is ignoring the persona → try few-shot, a stronger prompt, or a larger backbone; if it is still null, pivot the paper to the ceiling / "frozen VLMs can't personalize" finding (still publishable).

---

## 6. The synthetic audience (`src/audience/`) — the core method for C1

- `sample_personas(target_distribution, N)` → N persona cards drawn to match a target group (a slice spec) or the dataset marginal. The target distribution defines *which* group we simulate.
- `predict_group(image, personas)` → run the frozen judge once per persona → an empirical group distribution.
- `aggregate(distribution)` → the mean, the dispersion, the top complaints (mined from the rationales), and a between-group disagreement summary.
- **N sweep** (`sweep_N`) → recompute the group distribution at N = 1, 2, 5, 10, 20, 50; this feeds the C2 fidelity curve.
- **Caching** — cache judge outputs keyed by `(image_id, persona_hash, model_id, provider, prompt_version, decoding_params)`, so the panel, the N-sweep, and every slice are all assembled from a single query per (image, persona, model, prompt).

---

## 7. Stage 2 — Audience-guided editing (`src/eval/c4_editing.py`)

1. Take an input image, sample an audience, and aggregate its complaints and between-group disagreements.
2. The **judge, acting as proposer**, distills a single edit instruction (constrained short text).
3. The **serverless editor** (FLUX.1-Kontext via a provider route when available; otherwise a provider-backed fallback) produces K candidates (K = 4).
4. **Rerank using held-out signals only** — EditReward if serverless-accessible, otherwise EditReward-Bench labels or serverless scalar evaluators. *The proposer never reranks.*
5. Compare the conditions: `{no-feedback, generic-VLM-critique, EditReward-only, audience-aggregate, audience-targeted}`.

---

## 8. Experiments and metrics

Everything is implemented in `src/metrics/`, and every headline metric returns a **bootstrap 95% CI** (1000 draws, resampling over both raters and images).

### 8.1 C1 — group prediction (headline)

Compare the predicted group reaction distribution (the aggregate of N personas) against the observed group (~25–35 raters/image), pooled per slice.

| Metric | Baselines | Pass criterion |
|---|---|---|
| Wasserstein-1, KL, and ECE between the predicted and observed **group** distributions (pooled per slice) | no-persona aggregate; population-mean prior | lower distributional error than both, with a CI excluding 0 |
| **Between-group separation** — corr(predicted slice-to-slice gaps, observed gaps) | no-persona (≈ 0 separation) | predicted gaps track observed gaps |

Slices are age bucket × art familiarity × (nationality where available). A minimum slice size is enforced, and we produce a reliability diagram per slice.

### 8.2 C2 — why aggregation works

- **Aggregate-vs-individual gap** — for the same model, report the per-rater individual error (Spearman ρ, MAE, against the Exp-0 ceiling) alongside the C1 group error, to show individual error ≫ group error.
- **N-personas fidelity curve** — group distributional error vs. panel size N ∈ {1, 2, 5, 10, 20, 50}; we expect a monotone decrease and saturation, and we report the saturating N.
- **Warm-start reference (individual only)** — collaborative filtering (`surprise` / matrix factorization) on seen users, reported as the individual lower-bound reference, not as a group baseline.

### 8.3 C3 — generalization to generated images (Rapidata)

- Convert judge scores to pairwise predictions: `p̂(a ≻ b | slice) = σ(β · (ŝ_a − ŝ_b))`, fitting β on a validation split.
- Define **disagreement pairs** — pairs where the slice win-rates differ across the top slices by `|Δwinrate| > 0.2`, with `n_votes ≥ 30` in each.
- **Primary metric** — pairwise accuracy on the disagreement pairs, and `ΔAUC = AUC(slice-conditioned) − AUC(global-preference)`, reported on the LAPIS-nationality subset.
- **Controls** — (i) a no-persona prompt (separates the persona signal from the default judgment) and (ii) a global-preference baseline (separates the cross-cultural signal from universal quality).
- Also report the **real-to-generated gap** relative to C1.

### 8.4 C4 — editing usefulness

- **EditReward-Bench** — ranking agreement (accuracy, Kendall-τ) for each condition.
- **C4-core (slicing-independent)** — win-rate of audience-aggregate vs. single-judge vs. no-feedback, using EditReward or a serverless held-out evaluator if available, EditReward-Bench labels, plus an optional human study.
- **C4-targeted** — within-group win-rate, **contingent on C3 showing a disagreement-pair signal**. If between-group variance is null, we report C4-core only.
- **Secondary** — edit drift (DINO/CLIP-I identity similarity to the source).

### 8.5 Leakage diagnostics

- **Memorization probe** — prompt the judge to reproduce an ArtEmis caption/title from the image; report the hit rate.
- Compare seen vs. artist-disjoint performance and report the gap.

---

## 9. Statistical analysis plan (`preregistration.md`)

- **Primary statistical endpoints (fixed before test access):** (1) the C1 group-distribution Wasserstein vs. the no-persona aggregate; (2) C1 between-group separation; (3) the C2 saturating-N and the aggregate-vs-individual gap; (4) the C3 ΔAUC on disagreement pairs; (5) the C4-core win-rate vs. single-judge.
- **Confidence intervals** — 1000-sample bootstrap, clustered by rater and by image.
- **Multiple comparisons** — Holm correction across the subgroup/slice sweep (secondary analyses).
- **Significance** — report the effect size and CI, not just a p-value; a result "counts" only if its CI excludes the null *and* it sits within the Exp-0 ceiling.

---

## 10. Ablations (all frozen — these are the method's only knobs)

1. **Persona-card fields** — drop each field group and report the change in Wasserstein distance.
2. **Zero-shot vs. few-shot** — generic vs. retrieved exemplars, over a sweep of k.
3. **Self-consistency** — T samples vs. a single sample.
4. **Calibration** — with vs. without the post-hoc mapping.
5. **Aggregation vs. single judge**, and the **persona-sampling scheme** (how the panel is drawn) — core to C1.
6. **Serverless judge/provider family** — the primary pinned VLM vs. 2–4 additional Inference Providers VLMs that pass the week-1 audit. The desired open-weight families (Gemma/Qwen/Kimi/Phi/Nemotron) are included only if provider-callable.
7. **Reranker source** — EditReward vs. a scalar evaluator, for C4.
8. *(Optional, non-validated)* a peer-revision audience probe.

---

## 11. Reproducibility and artifacts

- Log every run with its `seed`, config hash, model ID, prompt version, and dataset version.
- Save `results/*.json` (metrics + CIs) and auto-generate the tables and figures.
- Release only license-safe outputs (respecting the PARA/LAPIS/Rapidata licenses; EVA is CC0 and unrestricted), and release the code — and, if permitted, the optional benchmark — at camera-ready.

---

## 12. Week-by-week build order (with go/no-go gates)

| Week | Build | Gate |
|---|---|---|
| 1 | Environment; license/nationality audit; Inference Providers model audit; download HF data and clone EVA; unify the parquet; build the splits; the frozen serverless judge wrapper | data loads, splits are valid, the primary provider/model is pinned |
| 2 | **Exp 0 ceiling**; the persona-card builder; prompt design; the **steerability gate** | persona ΔR² > 0 and steerability > 0 (else pivot) |
| 3 | Post-hoc calibration; **C1 group prediction + C2 aggregation gap/N-curve + baselines** → **MPR locks** | C1 beats the no-persona aggregate; the N-curve improves with N |
| 4 | **C3 on Rapidata** (zero-shot; disagreement ΔAUC + no-persona/global controls); leakage | the C3 ΔAUC CI excludes 0 (else report the null honestly) |
| 5 | The editing loop; **C4-core on EditReward-Bench**; the serverless cross-model/provider panel on C1 + steerability; few-shot/self-consistency ablations | C4-core wins vs. single-judge |
| 6 | Ablations; bias/fairness diagnostics; figures; writing (2–6 pp) | statistical endpoints reported with CIs |

**Stretch goals (only once the MPR is solid):** C4-targeted, the optional human study, and a second editor (SD-3.5).

---

## 13. First commands (smoke test, day 1)

```bash
export HF_JUDGE_MODEL="CohereLabs/aya-vision-32b:cohere"  # replace after the week-1 audit
python -m src.data.download --datasets artelingo richhf rapidata editreward eva
python -m src.data.build_unified         # → processed/ratings.parquet
python -m src.data.make_splits --seed 0  # → splits/
python -m src.models.judge --backend hf_provider --model "$HF_JUDGE_MODEL" --provider auto --smoke
python -m src.eval.exp0_ceiling          # → results/exp0.json
```

---

## Appendix A — Judge prompt and output schema

The same template is used for every backbone; only the provider request wrapper differs.

**System:**

```
You are simulating one viewer's reaction to an image. You are given the viewer's
profile and one image. Predict THIS viewer's reaction, not a general opinion.
Respond ONLY with JSON matching the schema. Scores are 0.0–1.0.
```

**User:** `[IMAGE]` + the persona card (from §2) + the schema:

```json
{"score": 0.0, "emotion": "awe|amusement|sadness|fear|disgust|anger|contentment|excitement|other",
 "willingness_share": 0.0, "difficulty": 0.0, "rationale": "<=25 words"}
```

- **Decoding:** `temperature = 0`, `max_tokens = 128`. Prefer the provider's JSON/schema mode if supported; otherwise use a tolerant parser with one retry, then NaN.
- **For art (LAPIS):** only `score` and `rationale` are scored (no structured fields).
- **Edit-instruction prompt (C4 proposer):** input = the aggregated complaints → output = one imperative edit ≤ 15 words, no rationale.

---

## Appendix B — Baselines (exact implementations)

| Baseline | What it tests | How |
|---|---|---|
| **No-persona aggregate (C1)** | does the persona matter for the group? | run the judge with an empty profile, aggregate N copies → a group baseline |
| **Population mean (C1)** | the group floor | the per-image mean of the reference-pool raters; constant across groups |
| **Single-judge (C2)** | the value of aggregation | one persona instead of the aggregated N (also the N = 1 point on the curve) |
| **Features + metadata regressor (C2 ref)** | non-VLM personalization | image features (frozen CLIP/DINOv2 from HF) ⊕ one-hot persona → GradientBoosting/MLP (`sklearn`) → score |
| **Collaborative filtering (C2 ref)** | warm-start individual reference | matrix factorization (`surprise` SVD/ALS) on user × image; **warm-start only** (cold-start cannot predict) |
| **Global preference (C3)** | is it just universal quality? | the mean win-rate ignoring the slice |
| **No-persona prompt (C3)** | does the persona matter? | the same frozen judge with an empty profile |
| **EditReward-only (C4)** | reward-guided editing | pick the edit maximizing EditReward if serverless-accessible; otherwise a serverless held-out evaluator / bench-label variant, with no audience critique |

*Note: the features-plus-metadata regressor and CF are classical non-VLM baselines (sklearn / `surprise`), fit on frozen embeddings or the rating matrix — no backbone weights are touched, consistent with the frozen-only rule. They exist to show that persona-prompting beats simple personalization; they are not part of the method.*

---

## Appendix C — Metric definitions (`src/metrics/`)

- **Spearman ρ** (per rater, then averaged): `scipy.stats.spearmanr`.
- **MAE:** the mean of |ŝ − s| on `score_norm`.
- **Wasserstein-1:** `scipy.stats.wasserstein_distance(pred_samples, obs_samples)` per slice, averaged.
- **KL(obs ‖ pred):** histogram both into `B = 10` shared bins on [0,1], add ε = 1e-6, then `sum(p · log(p/q))`.
- **ECE:** `M = 10` confidence bins; `Σ (|B_m|/N) · |acc(B_m) − conf(B_m)|` (for the pairwise/binary predictions).
- **ICC(2,k):** the two-way random-effects model via `pingouin.intraclass_corr` (or statsmodels variance components).
- **ΔAUC (C3):** `roc_auc_score(label_a_wins, p̂_slice) − roc_auc_score(label, p̂_global)` on the disagreement pairs.
- **Between-group separation (C1):** `pearsonr(predicted slice-pair gaps, observed slice-pair gaps)`.
- **Bootstrap CI:** resample units (images/raters for C1–C2, image-pairs for C3) with replacement ×1000 → the 2.5/97.5 percentiles. Cluster by rater *and* image where both apply.

---

## Appendix D — Example config (`configs/predict_persona.yaml`)

```yaml
backend: hf_provider
model: ${HF_JUDGE_MODEL}       # frozen serverless VLM, pinned after the week-1 audit
provider: auto                 # or an explicit provider, e.g. cohere
provider_policy: fastest       # fastest | cheapest | preferred | explicit provider
bill_to: null                  # optional HF org billing target
data:
  ratings: data/processed/ratings.parquet
  split: splits/cold_start.json
prompt:
  mode: zero_shot            # zero_shot | few_shot
  few_shot: {k: 0, retrieve: false, encoder: facebook/dinov2-base}
  self_consistency: {samples: 1, temperature: 0.0}
calibration: {method: isotonic, fit_on: splits/calib_real.json}   # maps scores only
eval: {regime: cold_start, slices: [age_bucket, art_exp, nationality], bootstrap: 1000}
```

---

## Appendix E — Serverless cost budget

Every model call is inference-only; there is no training cost and no local-GPU requirement for the MPR.

| Job | Cost driver | Estimate / control |
|---|---|---|
| Exp 0 | a CPU mixed model | minutes locally; no provider calls |
| Audience inference (C1/C2) | **N personas × images × provider models** | the dominant serverless bill; cache by `(image_id, persona_hash, model_id, provider, prompt_version)`; sweep N up to 50 from the cached calls |
| C3 on Rapidata | ~tens of thousands of images, one query per slice/persona design | reuse the cached per-image/persona scores across pairs/slices; cap the slices before querying |
| Editing (C4) | K candidates × inputs through the provider editor | limit inputs to ~200; cache the generated candidates and the evaluation calls |

**The rule:** query each image once per `(persona, model, provider, prompt)`, then assemble panels/slices/pairs from the cache — never re-run the VLM per pair.

---

## Appendix F — Deliverables → claim → paper artifact

| # | Output | Claim | Paper artifact |
|---|---|---|---|
| Exp 0 | variance fractions + ceiling | motivates aggregation | Fig 1 (variance bars) |
| C1 | group-distribution error vs. no-persona/population + between-group separation | C1 (headline) | Table 1 + Fig 2 (reliability per slice) |
| C2 | aggregate-vs-individual gap + N-personas curve | C2 (why it works) | Fig 3 (N-curve) + gap row |
| Steerability | corr(pred Δ, empirical Δ) | validity gate | Table 1 footnote |
| C3 | ΔAUC on disagreement pairs + no-persona/global controls | C3 | Table 2 |
| C4 | win-rates (core + targeted) + edit-drift | C4 | Table 3 + qualitative figure |
| Ablations | field / few-shot / self-consistency / calibration / aggregation / sampling / backbone | robustness | Table 4 |
| Bias | subgroup calibration error | ethics | Appendix |

---

## Appendix G — Optional human study protocol (only if pursued)

- **Platform:** Prolific or in-house, with consent and a de-identified persona intake (the PARA/LAPIS fields plus a Big-5 short form, e.g. BFI-10).
- **Part A (psychographic groups, extends C3):** ~300 generated images (FLUX.1-dev / SD-3.5), ≥ 10 raters per image; the endpoint is the pooled subgroup distributional error.
- **Part B (group-targeted, extends C4):** a forced choice over the edit conditions; the endpoint is the within-group win-rate.
- **Power:** size for the *pooled subgroup* endpoint (the primary one); pre-register the sample size from a pilot variance estimate; individual-level analysis is exploratory.
- **IRB:** a single protocol, submitted in week 1; if it slips, we ship on public data (the paper does not depend on this).
- **Ethics:** frame outputs as dataset-sampled distributions, not group essences; report over- and under-statement of group gaps.

---

*Plan complete. Start with the §13 smoke test, then follow §12 week by week, honoring the go/no-go gates.*
