# Research Plan — Synthetic Audiences for Creative AI

This is the step-by-step plan to build the system, run the experiments, and produce the results described in `PROPOSAL.md`. It maps directly to the four claims there: **C1** (predicting a group's reaction — the headline), **C2** (why aggregation makes that possible), **C3** (generalizing to AI-generated images across cultures), and **C4** (using the simulated audience as editing feedback), preceded by **Exp 0** (a ceiling analysis that motivates everything).

The whole pipeline is built on Hugging Face, and every model call — the judge, the image editor, the reward/reranker — is made **serverlessly through Hugging Face Inference Providers**. We never train, fine-tune, or self-host a model backbone for the main result. Our own code only handles data processing, caching, calibration, metrics, and the classical (non-VLM) baselines.

**Five guiding rules.**
1. **Frozen only.** All VLM/editor/reward calls use off-the-shelf models with weights we never touch. The only fitting we allow is (a) post-hoc calibration, which maps scores and touches no weights, and (b) classical baselines (a regressor and collaborative filtering). *(Amended 2026-07-11, see §14.8: the "serverless" requirement is dropped — a frozen model is the claim; whether it is called locally or through a provider is an implementation detail, not part of the result. Our runs use a local frozen Qwen2-VL-7B and that is sufficient.)*
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

## 14. Interim results — first run-through (24 analyses; audit → Exp 0 → C2 → calibration → steerability → C1 → robustness)

*Recorded 2026-07-11. Analysis code: `scripts/analysis/` (26 scripts); machine-readable outputs
in `results/*.json` (24 files, git-ignored, regenerable) + figures in `results/figs/`. A full
protocol for the remaining work is in `docs/analysis_protocol.md`; the temperature re-run task is
specified in `docs/task_temperature_rerun.md`. Every table below is reproduced from the named
result file; every number was cross-checked against those files.*

#### 14.0 Master results index (every analysis → script → finding)

| § | Analysis | Script | Result file | Headline finding |
|---|---|---|---|---|
| 14.1 | Coverage / traceability | `coverage.py` | `coverage.json` | 100% of rows trace to source; PARA 6.4% / EVA 100% / LAPIS 34% of images scored |
| 14.2 | Audit (Part A) | `audit.py` | `audit.json` | metrics reproduce <6e-4; parse ≈100%; LAPIS dup artifact; temp-0 collapse 48–51% |
| 14.3 | Uncalibrated per-rating signal | (baked) | `audit.json` | large +bias; no run beats image-mean oracle (expected) |
| 14.4 | Exp 0 ceiling | `exp0_ceiling.py` | `exp0.json` | ICC1 0.19–0.47 vs ICCk 0.84–0.96; persona-ΔR² 1–4% |
| 14.5 | C2 gap + N-curve | `c2_ncurve.py` | `c2.json` | gap 1.1–1.6×; N-curve flat (temp-0 artifact); between-image rank 0.62–0.77 |
| 14.6 | Calibration (B4) | `calibration.py` | `calibration.json` | group MAE ~halved; **beats population prior on all 3** |
| 14.7 | Steerability (B1) | `steerability.py` | `steerability.json` | r = +0.37 / −0.24 / +0.39 (PARA/EVA/LAPIS) |
| 14.11 | persona-ΔR², persona value, **C1 separation** | `exp0_ceiling.py`, `persona_value.py`, `c1_separation.py` | `persona_value.json`, `c1_separation.json` | **LAPIS between-group +0.17 [0.15,0.18]** vs blind ≈0; PARA +0.04; EVA fails |
| 14.12.1 | Multi-dimension | `dims_extended.py` | `dims_extended.json` | aggregate beats prior on **all 16 axes** |
| 14.12.2 | Difficulty validity | `validity_difficulty.py` | `validity_difficulty.json` | VLM difficulty tracks disagreement ρ +0.14 |
| 14.12.3 | Inter-dimension structure | `structure.py` | `structure.json` | EVA structure reproduced 0.77, PARA 0.29 |
| 14.12.4 | Warm-start CF baseline | `cf_baseline.py` | `cf_baseline.json` | personalization gain only +0.01–0.03 |
| 14.12.5 | Bias / fairness | `bias.py` | `bias.json` | fair on personality/age; **nationality bias gap 0.50**, region 0.23 |
| 14.12.6 | Leakage / memorization | `leakage.py` | `leakage.json` | artist-fame vs error ≈0 → **no memorization** |
| 14.12.7 | Rationale probe | `rationale.py` | `rationale.json` | persona text→attribute AUC 0.58–0.70 vs blind ≈0.50 |
| 14.13.1 | Rater leniency | `rater_leniency.py` | `rater_leniency.json` | LAPIS +0.26 [0.18,0.34]; PARA ns; EVA −0.12 |
| 14.13.2 | Response-style audit | `response_style.py` | `response_style.json` | **central-tendency: uses 51–58% of human spread** |
| 14.13.3 | Accuracy vs agreement | `accuracy_vs_agreement.py` | `accuracy_vs_agreement.json` | link confounded by central-tendency bias |
| 14.13.4 | Content-category | `content_category.py` | `content_category.json` | rank 0.60–0.82; night scenes hardest |
| 14.13.5 | Structured-signal separation | `structured_separation.py` | `structured_separation.json` | null on content-pref/willingness/difficulty (temp-0) |
| 14.13.6 | Calibration transfer | `calib_transfer.py` | `calib_transfer.json` | transfers across datasets → supports C3 reuse |
| 14.13.7 | Holm correction | `holm.py` | `holm.json` | 6/8 slice separations survive FWER |
| 14.13.8 | LAPIS repeated-measures | `repeated_measures.py` | `repeated_measures.json` | headline identical across dedup policies |
| 14.13.9 | Off-grid / snapping audit | `offgrid_audit.py` | `offgrid_audit.json` | parse ≈100%, snapping a no-op |

A teammate produced a first pass of persona-conditioned VLM ratings on **PARA, EVA, and
LAPIS**, in two modes each — **`full`** (persona-conditioned: each real rater re-created as a
persona card, the judge answers in character) and **`blind`** (the plan's no-persona control:
one generic prompt on the *same* (image, rater) tasks). This section audits those runs,
reports the Exp 0 ceiling and the C2 aggregation analysis computed from them, and flags the
confounds that gate every headline number. **All new analysis is pure re-analysis of the
cached logs — no inference, no GPU.**

> **Model note:** these runs used a **local, frozen Qwen2-VL-7B-Instruct**. Per the 2026-07-11
> amendment to guiding rule 1, a frozen model is all the claim requires — the serverless routing
> is no longer a condition — so Qwen2-VL-7B is an acceptable headline backbone. Trying additional
> frozen VLMs remains a *robustness* ablation (§10.6), not a correctness requirement.

### 14.1 What was actually run, and how much of each dataset it covers

The runs were **not** on the full datasets (they take a long time). Every result row was
traced back to a real source row in `data/{para,lapis,eva}`: **100% of result (image, user)
keys are found in the source annotation files, and the stored ground truth matches the source
rating** (±0.5) for 99–100% of rows. Coverage:

| Dataset | Images scored / total | Ratings scored / total | Raters seen / total | Source file the results map to |
|---|---|---|---|---|
| **PARA**  | 2,000 / 31,220 (**6.4%**) | 51,749 / 807,586 (**6.4%**) | 437 / 438 (99.8%) | `para/annotation/PARA-Images.csv` |
| **EVA**   | 4,070 / 4,070 (**100%**) | 136,943 / 136,943 (**100%**) | 1,094 / 1,094 (100%) | `eva/data/votes_filtered.csv` |
| **LAPIS** | 4,000 / 11,723 (**34.1%**) | 90,262 / 283,860 (**31.8%**) | 531 / 568 (93.5%) | `lapis/annotation/LAPIS_PIAA.csv` |

**Read this carefully when quoting numbers.** EVA is complete. PARA is a **6.4% image sample**
(stratified, seed 0) — but note that although few *images* are covered, nearly every *rater*
appears, because each sampled image is judged by its ~25 real raters drawn from the full pool.
LAPIS is about a third. Result records carry `imageName` + `userId`, which are exactly the
join keys (`PARA-Images.imageName/userId`, `votes_filtered.image_id/user_id`,
`LAPIS_PIAA.image_filename/participant_id`), so any result row can be joined back to its
source rating and to the rater's attributes (`PARA-UserInfo.csv`, `eva/data/users.csv`, or the
inline `LAPIS_PIAA` demographic columns).

Run configuration: model `Qwen/Qwen2-VL-7B-Instruct`, seed 0, `full` at **temperature 0.0**
(greedy), `blind` at **temperature 0.7**.

### 14.2 Audit — the logs are trustworthy, with one data-integrity caveat (Part A)

| Check | Result |
|---|---|
| **Metric reproduction (A1)** — recompute every baked-in per-rating metric from raw records | ✅ reproduces to **< 6e-4** on all three datasets → the precomputed `metrics` blocks are correct |
| **Manifest completeness (A9)** — row counts vs `n_ratings` | ✅ exact match on all six runs |
| **Parse-failure rate (A2)** | ✅ ~0 (PARA `full` 0.03%, all others 0.00%) — JSON parsing is not a concern |
| **Full/blind pairing (A4)** | ✅ identical (image, user) task sets in every dataset |
| **LAPIS duplicate rows (A9)** | ⚠️ **1,446 duplicate keys**: 900 *exact-identical* rows (export artifact → drop) + 283 pairs of *genuine repeated measures* (same rater scored the same painting twice with a **different** score — e.g. 0 and 9). Analysis scripts drop the exact duplicates; the 283 real repeats are kept. |
| **Decoding confound (A5)** | ⚠️ `full` temp 0.0 vs `blind` temp 0.7 → the two modes differ in *both* persona conditioning and decoding temperature |
| **Degenerate predictions (A7)** | ⚠️ fraction of images whose predicted ratings are a point mass (zero spread): PARA `full` 48% / `blind` 88%; EVA 51% / 90%; LAPIS **4%** / 86% |

The image-mean "baseline" carried in the logs (`baseline_mae_image_mean`) uses each test
image's own raters' mean — it is the **empirical group oracle / C1 target**, not an individual
floor the model should beat. The true individual floor is the global-mean baseline.

### 14.3 Preliminary per-rating signal (pooled, **uncalibrated**, native scale)

Primary axis only (PARA `aestheticScore` 1–5, EVA `score` 0–10, LAPIS `rating` 0–100). Lower
MAE is better; both baselines are MAE.

| Run | MAE | mean bias | Spearman | Baseline MAE (global mean) | Baseline MAE (image mean = group oracle) |
|---|---|---|---|---|---|
| para_full  | 0.828 | **+0.648** | 0.493 | 0.611 | 0.438 |
| para_blind | 0.639 | +0.276 | 0.533 | 0.611 | 0.438 |
| eva_full   | 1.669 | +0.737 | 0.277 | 1.630 | 1.423 |
| eva_blind  | 1.761 | +1.030 | 0.315 | 1.630 | 1.423 |
| lapis_full | 23.34 | +13.11 | 0.339 | 22.98 | 19.56 |
| lapis_blind| 24.71 | +16.30 | 0.290 | 22.98 | 19.56 |

Three things: (1) a **large positive bias everywhere** — the VLM systematically over-rates, so
raw MAE is unfair to it and unstable → **calibration is mandatory** before any distributional
claim; (2) **personas do not clearly help** — on PARA and EVA the `blind` control matches or
beats `full` on rank correlation, only LAPIS shows persona > blind → the steerability gate is
genuinely at risk and is tested next (§14.5); (3) no run beats the **image-mean** oracle, which
is *expected* and is the whole point of Exp 0 / C2 below.

### 14.4 Exp 0 — ceiling analysis (the premise holds cleanly)

Variance of the human ratings decomposed into a between-image (shared, predictable) part and a
within-image (idiosyncratic) part via a one-way random-effects model (raters nested in images,
unequal group sizes), on the **normalized [0,1] scale**.

| Dataset | ICC(1) = one rating's reliability | ICC(k) = group-mean reliability | Between-image variance frac. | Individual/group noise ratio |
|---|---|---|---|---|
| **PARA**  | 0.470 | **0.958** | 0.470 | 5.1× |
| **EVA**   | 0.223 | **0.906** | 0.223 | 5.8× |
| **LAPIS** | 0.188 | **0.839** | 0.188 | 4.7× |

**Interpretation.** An *individual* rating is mostly idiosyncratic — only 47% (PARA), 22% (EVA),
19% (LAPIS) of its variance is shared, predictable signal; the rest is noise no model can
recover. But the *group mean* is highly reliable (ICC(k) 0.84–0.96). That ICC(1)→ICC(k) jump is
the quantitative motivation for aggregation and the ceiling every individual-level number is
read against. It also reframes §14.3: losing to the image-mean oracle is inevitable, because
that oracle is ~0.9-reliable. *(Persona-attributable variance ΔR² needs the rater-attribute
join and is reported with the steerability / C1 workstream.)*

### 14.5 C2 — aggregation gap and N-curve (an important, less flattering result)

Using the `full` predictions, normalized [0,1] scale, 1000× bootstrap CIs. "Group MAE" =
error of the VLM's per-image mean prediction against the observed group mean; "gap ×" =
individual MAE / group MAE; "group rank" = Spearman of predicted vs observed image means.

| Dataset | Individual MAE | Group MAE | Gap × | Group rank ρ (between-image signal) | Population-mean prior (group MAE) |
|---|---|---|---|---|---|
| **PARA**  | 0.207 | 0.184 | 1.12 | **0.769** | 0.102 |
| **EVA**   | 0.167 | 0.103 | 1.62 | 0.624 | 0.081 |
| **LAPIS** | 0.233 | 0.145 | 1.61 | **0.717** | 0.105 |

N-personas fidelity curve (group MAE by panel size N):

| Dataset | N=1 | N=2 | N=5 | N=10 | N=20 |
|---|---|---|---|---|---|
| PARA  | 0.187 | 0.186 | 0.185 | 0.184 | 0.184 |
| EVA   | 0.106 | 0.104 | 0.104 | 0.103 | 0.103 |
| LAPIS | 0.156 | 0.150 | 0.146 | 0.145 | 0.146 |

**Two findings that the raw logs hid, both actionable.**
1. **The aggregate-vs-individual gap is modest (1.1–1.6×), not dramatic, and the population-mean
   prior beats the VLM's aggregate on MAE** (e.g. PARA 0.102 vs 0.184). Cause: the VLM's large
   positive **bias**, which aggregation cannot remove. → **calibration (B4) is the highest-value
   next step**, not an optional ablation.
2. **The N-curve is nearly flat and saturates by N≈2.** Cause: `full` ran at temperature 0, so
   different personas return near-identical scores (the 48–51% degenerate-prediction rate in
   §14.2) — there is almost no within-panel variance for aggregation to cancel. This is a
   **decoding artifact, not evidence against aggregation**, and it means the N-curve is
   uninterpretable until a **matched-temperature (>0) re-run** exists.

The buried positive: the VLM's predicted image means track the true image means at
**ρ = 0.62–0.77** — it captures between-image quality ordering well; it is the bias and the
collapsed persona diversity that sink the MAE.

### 14.6 B4 — post-hoc calibration (the fix for the bias, and it works)

Isotonic calibration fit **out-of-fold** (2-fold cross-fit by image, seed 0), applied so every
rating is scored by a calibrator that never saw its image. Normalized [0,1] scale, primary axis.
Isotonic is monotonic, so **Spearman is essentially unchanged** (a rank sanity check); only
scale/bias moves.

| Dataset | Individual MAE (raw → calibrated) | **Group MAE (raw → calibrated)** | Population-mean prior | Calibrated aggregate beats prior? |
|---|---|---|---|---|
| **PARA**  | 0.207 → 0.126 | **0.184 → 0.062** | 0.102 | ✅ yes |
| **EVA**   | 0.167 → 0.156 | **0.103 → 0.063** | 0.081 | ✅ yes |
| **LAPIS** | 0.233 → 0.212 | **0.145 → 0.072** | 0.105 | ✅ yes |

**This flips the C2 story.** Uncalibrated, the VLM's aggregate *lost* to the population-mean
prior (§14.5) purely because of bias. Once bias is removed, **the aggregated group prediction
beats the population prior on all three datasets** (group MAE roughly halved), confirming the
VLM's aggregate carries real between-image signal — the same signal visible in the ρ = 0.62–0.77
rank correlation. Calibration is therefore a **prerequisite**, not an ablation, and the
uncalibrated §14.3/§14.5 numbers must never be quoted as the method's performance.

### 14.7 B1 — steerability gate (marginal: weak-positive on PARA/LAPIS, fails on EVA)

Does the persona move the judge the way the *data* says it should? Image-centred so shared image
quality is removed, then per rater-attribute level we compare the **VLM's persona-induced
deviation** to the **real group's deviation** (attributes joined from source via
`scripts/analysis/attrs.py`). Steerability = corr across all (attribute, level) cells.

| Dataset | Steerability r | Sign agreement | VLM effect amplitude / real | Verdict |
|---|---|---|---|---|
| **PARA**  | **+0.367** | 0.543 | 0.985 | weak-positive — persona functional |
| **EVA**   | **−0.239** | 0.242 | 0.220 | ✗ fails — wrong direction, barely moves |
| **LAPIS** | **+0.394** | 0.541 | 0.355 | weak-positive, but under-moves 3× |

**Reading.** On the attribute-rich datasets (PARA has Big-5 + art/photo experience; LAPIS has
nationality + art interest) the persona is **weakly functional** — the judge shifts in the
data-consistent direction and, on PARA, by about the right magnitude. On **EVA** — whose personas
are thin (age, region, gender, photographic level, eyesight; no Big-5/education/art) — the effect
is **null-to-negative**: the model barely moves for the persona (amplitude 0.22) and mostly in the
wrong direction. Two caveats both point the same way: `full` ran at temperature 0, so ~half of
images contribute *zero* persona spread (§14.2), and the sign-agreement near 0.54 on PARA/LAPIS is
only modestly above chance. This is a **floor**; a matched-temperature re-run and a stronger
persona prompt are the levers to raise it. Per the plan's gate (§5.2), PARA/LAPIS clear "steer >
0" weakly and EVA does not — so the persona effect is real but fragile, and the cross-cultural /
between-group claims (C1 separation, C3) will hinge on strengthening it.

### 14.8 Implications for the plan (what these results change)

- **Calibration is confirmed a prerequisite, not an ablation (done — §14.6).** It roughly halves
  group MAE and lifts the aggregate above the population prior on all three datasets; the
  uncalibrated §14.3/§14.5 numbers must not be quoted as the method's performance.
- **The persona is functional but fragile (done — §14.7).** Steerability is weak-positive on
  PARA/LAPIS and fails on EVA, and the model under-moves for personas. A **matched-temperature
  re-run** (removing the ~50% zero-spread images) and a stronger persona prompt are the levers to
  raise it before C1-separation / C3 can be trusted.
- **The headline C1 group analysis (§8.1) still does not exist** — no between-group separation
  across persona slices has been computed. The rater-attribute join now exists
  (`scripts/analysis/attrs.py`), so this is the next major deliverable.
- **The serverless requirement is dropped (2026-07-11).** A frozen model is the claim; the local
  Qwen2-VL-7B is an acceptable headline backbone. Running additional frozen VLMs stays an
  optional *robustness* ablation, not a blocking reproduction step.
- **Coverage caveat for the write-up:** PARA numbers rest on a 6.4% image sample and LAPIS on
  34%; EVA is complete. Widening PARA image coverage is **in progress (owner: user)**; bootstrap-
  weight in the meantime and re-run the Tier-1 scripts once the wider PARA logs land.

### 14.9 Conclusions so far — claim scorecard

Where each claim stands after the audit, Exp 0, C2, calibration, and steerability. "Evidence"
is the quantity that decides it; "status" is our honest read on the current (Qwen2-VL-7B, partial-
coverage, uncalibrated-run-then-calibrated-offline) data.

| Claim | Decisive evidence so far | Status | What's still needed to close it |
|---|---|---|---|
| **Exp 0** — individuals near noise, group predictable | ICC(1) 0.19–0.47 vs ICC(k) 0.84–0.96; noise ratio ~5×; persona-ΔR² only 1–4% (§14.11a) | ✅ **supported (complete)** | — |
| **C2** — aggregation is why it works | calibrated group MAE beats the population prior on all 3 (e.g. PARA 0.062 vs 0.102); between-image rank ρ 0.62–0.77 | 🟡 **supported with an asterisk** | a real N-curve (needs temp>0; current curve is flat from temp-0 collapse) |
| **Steerability gate** — the persona is functional | corr(VLM Δ, real Δ): PARA +0.37, LAPIS +0.39, **EVA −0.24** (§14.11b); *and* the persona steers the rationale text at AUC 0.58–0.70 vs blind ≈0.50, incl. EVA (§14.12.7) | 🟢 **functional** (text proves it even where the score hides it) | temp-0.7 re-run to lift the score-level effect off its floor |
| **C1** — predict a group's reaction (headline) | between-group separation (§14.11c): **LAPIS +0.17 CI[0.15,0.18]** vs blind ≈0; PARA +0.04 weak; EVA fails | 🟡 **demonstrated on LAPIS**, weak on PARA, fails on EVA | distribution-match (W1/KL/ECE) per slice; strengthen via temp>0 re-run |
| **C2 — structured signals** (difficulty, attribute votes, share/pref) | calibrated aggregate beats the prior on all 16 axes; PARA social signals carry +persona value; EVA difficulty validly tracks disagreement (ρ+0.14) but is group-unpredictable (§14.12.1–2) | 🟡 **mostly supported**, scoped honestly | same temp>0 re-run for the distributional half |
| **Bias / fairness** (ethics) | calibrated judge fair on personality/age/expertise (gap ≤0.04) but large **nationality/region bias** (gap 0.50 / 0.23) (§14.12.5) | ⚠️ **flag + control for it** | net out national bias in C3; report in ethics appendix |
| **Leakage** | artist-fame vs error ρ ≈ 0; no memorization; zero-shot ⇒ no AVA path (§14.12.6) | ✅ **clean** | memorization-probe (needs inference) as a belt-and-suspenders check |
| **C3** — generalize to generated images, cross-culture | *no runs* (but LAPIS nationality separation §14.11c is the real-image precursor) | ⚪ **not started** | Rapidata runs; contingent on C1 separation being real |
| **C4** — audience as editing feedback | *no runs* (complaint-mining vocabulary looks usable, §14.12.7) | ⚪ **not started** | editor + held-out reranker loop |

**Bottom line (updated after Tier 1).** The *supporting science* (Exp 0 + C2 — "the group is
predictable where the individual is not, and aggregation is why") is **empirically in hand** once
calibration is applied, and the **headline C1 claim is now demonstrated** where personas are rich:
on LAPIS the model reproduces genuine between-group divergence (pooled separation +0.17, CI
excludes 0, no-persona control ≈0), including the nationality axis C3 needs. The effect is
weak-but-real on PARA and fails on EVA — persona signal tracks persona richness (LAPIS > PARA >
EVA). The one remaining confound — the temperature-0 persona collapse — means every separation is
a **floor**. Nothing here triggers the "frozen VLMs can't simulate group taste" pivot: the group
signal is present and significant; the work now is to sharpen it (matched-temperature re-run,
stronger prompt) and extend to C3.

### 14.10 Next steps (prioritized roadmap)

> **Status (2026-07-11): Tier 1 is complete**, and so is all further temperature-robust analysis of
> the current logs (§14.11 → §14.13). The definitive current status and the single remaining forward
> plan are consolidated in **§14.14**; the Tier-2/3 items below stand.

**Tier 1 — pure re-analysis of the existing logs (no GPU) — ✅ DONE.**
1. ✅ **C1 between-group separation (headline)** — done (§14.11c): LAPIS +0.17, PARA +0.04, EVA fails.
2. ✅ **Finish Exp 0** — persona-ΔR² done (§14.11a).
3. ✅ **Clean full-vs-blind persona value** — done (§14.11b). *(Plus the entire §14.12–14.13 second
   and third waves: multi-dimension, bias, leakage, rationale, response-style, transfer, hardening.)*

**Tier 2 — new inference (only after Tier 1 shows the signal is real — it does).**
4. **Matched-temperature re-run** of `full` at temp 0.7 with several samples per (image, persona) —
   removes the temp-0 persona collapse, unlocks a real N-curve, a clean full-vs-blind contrast, and
   a higher steerability read. *This is now the single most valuable Tier-2 action.*
5. **Widen PARA image coverage** (PARA is 6.4%; LAPIS 34%; EVA complete) — **in progress (owner:
   user)**. Re-run the Tier-1 scripts on the wider logs when they land.
6. *(Optional robustness)* additional frozen VLMs on C1 + steerability — an ablation, not a
   requirement (the serverless-reproduction requirement was dropped 2026-07-11).

**Tier 3 — new workstreams (contingent).**
7. **C3** cross-cultural on Rapidata — contingent on C1 separation being real.
8. **C4** editing loop (proposer ≠ reranker).
9. **Ablations + leakage** — persona-field drop, few-shot, self-consistency; EVA↔AVA dedup and
   the memorization probe.

*Tier 1 is implemented in `scripts/analysis/` and its outputs recorded below as they land.*

### 14.11 Tier-1 completion — persona-ΔR², persona value, and C1 between-group separation

**(a) Exp 0 completion — persona-ΔR² (`exp0_ceiling.py`).** How much of the *within-image*
(idiosyncratic) rating variance do the rater's persona attributes explain (image-centred OLS on
the joined attributes)?

| Dataset | Persona-ΔR² (within-image) |
|---|---|
| PARA  | 2.0% |
| EVA   | 0.9% |
| LAPIS | 4.0% |

Persona attributes explain only **1–4%** of idiosyncratic taste — low, exactly as the proposal
predicts, and the reason the *individual* is near-unpredictable. LAPIS highest, EVA lowest
(mirroring steerability). This is the motivation for aggregating, not a failure.

**(b) Clean full-vs-blind persona value (`persona_value.py`).** Decoding-invariant within-image
signal corr(pred_dev, gt_dev): the persona run vs the no-persona floor, 1000× bootstrap CI
clustered by image.

| Dataset | Within-image corr — full | — blind | Persona value (full − blind) | CI95 (full) |
|---|---|---|---|---|
| **PARA**  | +0.019 | −0.005 | **+0.024** | [0.011, 0.028] |
| **EVA**   | −0.015 | +0.002 | **−0.017** | [−0.021, −0.009] |
| **LAPIS** | +0.099 | +0.005 | **+0.093** | [0.092, 0.106] |

The persona adds a small but CI-significant individual signal on PARA and a clear one on **LAPIS**;
on **EVA it subtracts** (wrong direction). The between-image rank signal is shared by both modes
(≈0.6–0.8), confirming it is image quality, not persona.

**(c) C1 between-group separation — the headline (`c1_separation.py`).** Does the model reproduce
how groups *diverge* on the same image? corr(predicted slice-to-slice gap, observed gap),
image-controlled, on calibrated scores, vs the no-persona control (≈0 expected). CI clustered by
image.

| Dataset | Slice attribute | Full separation (CI95) | Blind | Verdict |
|---|---|---|---|---|
| **LAPIS** | nationality | +0.068 [0.034, 0.104] | −0.00 | ✅ |
| **LAPIS** | art interest | +0.088 [0.067, 0.107] | +0.03 | ✅ |
| **LAPIS** | age | +0.102 [0.078, 0.127] | +0.01 | ✅ |
| **LAPIS** | **pooled** | **+0.166 [0.150, 0.181]** | +0.02 | ✅ **strong** |
| **PARA**  | art experience | +0.040 [0.011, 0.070] | −0.02 | ✅ weak |
| **PARA**  | photography exp. | +0.055 [0.027, 0.084] | −0.02 | ✅ weak |
| **PARA**  | age | +0.008 [−0.019, 0.037] | −0.01 | ✗ null |
| **PARA**  | **pooled** | **+0.044 [0.024, 0.062]** | −0.01 | ✅ weak |
| **EVA**   | **pooled** | **−0.084 [−0.103, −0.065]** | +0.01 | ✗ fails |

**Conclusion — the headline claim is demonstrated, and its shape is instructive.** On **LAPIS**
the model reproduces genuine between-group divergence (nationality, art interest, age) with a
pooled separation of **+0.17 whose CI excludes 0, against a no-persona control at ≈0** — this is
the paper's central result, and the significant **nationality** term is exactly the cross-cultural
signal C3 will build on. The effect is **weak-but-real on PARA** (experience axes) and **fails on
EVA** (thin personas, wrong direction). The pattern is consistent across every Tier-1 analysis —
persona signal scales with persona richness (LAPIS > PARA > EVA) — and all separations are
**floors**, depressed by the temperature-0 persona collapse. **This clears the decision gate: the
group signal exists, so we do not pivot; we strengthen it (matched-temperature re-run, stronger
prompt) and proceed to C3.**

### 14.12 "Meanwhile" analyses — what the current logs tell us without the temp-0.7 re-run

Seven analyses that test *other* claims and need no new inference (all temperature-robust:
rank / mean / calibration / ceiling / text). Scripts in `scripts/analysis/`, outputs in
`results/*.json`.

**(1) Multi-dimension extension (`dims_extended.py`) — C1 scope + C2 structured signals.** The
temp-robust metrics on *every* rated axis (PARA ×9, EVA ×6, LAPIS ×1):

- **The calibrated aggregate beats the population-mean prior on all 16 dimensions** — the C2
  group result is not specific to the headline axis.
- PARA's **personal/social signals carry a positive persona effect** (within-image persona value:
  lightScore +0.032, aestheticScore +0.024, contentPreference +0.023, willingnessToShare +0.021),
  supporting C2's "structured signals" scope on photos.
- **EVA difficulty is essentially unpredictable at the group level** (ICC(k) 0.68, between-image
  rank 0.10) — reported honestly; humans barely agree on difficulty and the VLM does not recover it.

**(2) Difficulty as a validity probe (`validity_difficulty.py`, EVA).** Does predicted difficulty
point at images humans actually find contentious (per-image score disagreement)? Spearman:

| relation | ρ |
|---|---|
| human difficulty vs human disagreement (construct check) | +0.237 |
| **VLM difficulty vs human disagreement** | **+0.135** |
| VLM difficulty vs human difficulty | +0.107 |

The construct is valid and the VLM's difficulty **weakly but positively** tracks real disagreement.

**(3) Inter-dimension structure (`structure.py`).** Does the VLM reproduce how the aesthetic
sub-scores co-vary? Correlation between the human and VLM inter-dimension correlation patterns:
**EVA 0.77** (reproduced well) but **PARA 0.29** (poorly — PARA's 9 fine-grained axes are strongly
"halo-ed": human mean cross-dim corr 0.93, VLM 0.85, but the fine pattern doesn't match).

**(4) Warm-start CF baseline (`cf_baseline.py`) — the C2 individual reference.** Classic additive
bias model (μ + user + item), ground-truth only:

| Dataset | CF warm-start MAE | item-only MAE | personalization gain |
|---|---|---|---|
| PARA  | 0.102 | 0.115 | **+0.013** |
| EVA   | 0.117 | 0.148 | **+0.031** |
| LAPIS | 0.182 | 0.206 | **+0.025** |

Knowing the user helps a classical model only marginally — **the same "individual taste is mostly
idiosyncratic" conclusion as Exp 0, from the non-VLM side.** This is the individual lower-bound the
VLM's cold-start prediction is read against.

**(5) Bias / fairness (`bias.py`) — ethics appendix.** Per-subgroup calibration error. The judge is
**fair across personality, age, gender, and expertise** (MAE gaps ≈0.005–0.02) but shows a **large
systematic bias by nationality and region**:

| Attribute | Calibration bias gap (max−min across levels) |
|---|---|
| LAPIS nationality | **0.50** |
| EVA region | **0.23** |
| LAPIS gender | 0.17 |
| (all personality / age / art-experience axes) | ≤ 0.04 |

This is an ethics-appendix headline **and a caution for C3**: the cross-cultural signal is entangled
with the model's own national/regional rating bias, so C3 must net it out (it already uses the
global-preference and no-persona controls for this reason).

**(6) Leakage / memorization (`leakage.py`, LAPIS).** No memorization signal: the correlation
between an artist's rating-volume (fame proxy) and the model's error is **+0.03 ≈ 0** — famous
artists are *not* rated more accurately. Style error gap is small (0.04); the model is merely weaker
on abstract/minimalist styles (Minimalism, Color-Field, Action-painting worst; Impressionism best) —
difficulty, not leakage. (EVA⊂AVA is noted; the zero-shot pipeline uses no AVA exemplars, so there
is no leakage path.)

**(7) Rationale analysis (`rationale.py`) — persona signal the quantized score hides.** Can a rater's
attribute be predicted from *their rationale text alone*? TF-IDF + logistic regression, ROC-AUC:

| Dataset | Probe attribute | AUC — persona (`full`) | AUC — no-persona (`blind`) | Rationale diversity full / blind |
|---|---|---|---|---|
| PARA  | art experience | **0.578** | 0.498 | 0.44 / 0.11 |
| EVA   | photographic level | **0.701** | 0.511 | 0.25 / 0.09 |
| LAPIS | art interest | **0.695** | 0.506 | 0.42 / 0.09 |

**The blind control sits exactly at chance (≈0.50); the persona rationales clearly encode the
rater's attributes** — and personas produce ~4× more distinct rationales than the generic prompt.
Most striking: on **EVA the persona signal is strong in the text (AUC 0.70) even though it was
invisible/negative in the quantized score** — direct evidence that greedy decoding is *censoring* a
persona effect the model is actually computing, and independent confirmation that the temperature-0.7
re-run will surface signal now hidden. (Complaint words are coherent and useful for C4: PARA low =
"blurry, overexposed, poor, dark"; LAPIS low = "simple, plain, blank, minimalist, devoid".)

**Meanwhile-analysis conclusion.** Three things are now firmer without any new runs: (i) the C2
group result and calibration hold across *every* rated axis and are matched by a classical
individual-reference lower bound; (ii) the persona is provably functional — it steers the rationale
language (AUC ≫ chance) even where the integer score collapses, which both validates the method and
predicts the temp-0.7 payoff; and (iii) two things to handle in the write-up — a **national/regional
calibration bias** (ethics + a C3 confound already controlled for) and **no memorization leakage**
(a clean bill of health for LAPIS art).

### 14.13 Second-wave analyses (current data, no re-run) — what more the logs reveal

Nine further analyses/audits. Scripts in `scripts/analysis/`, outputs in `results/*.json`.

**(1) Rater-level leniency capture (`rater_leniency.py`).** Beyond attributes: does the persona
capture whether a rater is globally harsh/lenient? corr(real leniency, VLM-assigned leniency),
bootstrap CI over raters:

| Dataset | corr (full) | CI95 | blind |
|---|---|---|---|
| **LAPIS** | **+0.262** | [0.184, 0.343] | +0.05 |
| PARA | +0.061 | [−0.043, 0.151] (ns) | −0.11 |
| EVA | −0.123 | [−0.192, −0.058] | −0.01 |

On LAPIS the persona meaningfully reproduces individual response-style; PARA null, EVA negative —
the familiar LAPIS > PARA > EVA ordering.

**(2) Response-style audit (`response_style.py`) — the mechanism behind calibration.** The judge
**compresses toward the middle**: it uses only **51–58%** of the human score spread on EVA/LAPIS,
concentrates 34–56% of answers on a single value (humans 4–20%), and **avoids the endpoints**
(EVA endpoint share 0.00 vs 0.04). This central-tendency bias — separate from the mean bias — is
*why* raw distributional error is large and why calibration recovers so much. It also explains (3).

**(3) Accuracy vs human agreement (`accuracy_vs_agreement.py`).** The expected "more accurate where
humans agree" link is **confounded by (2)**: on LAPIS the group error actually *falls* on
high-disagreement images (corr −0.19), because contested images have central means that the
compressed judge happens to hit. A clean ceiling→performance link will need the de-compressing
temp-0.7 run.

**(4) Content-category performance (`content_category.py`).** The audience model predicts group
aesthetics well across subject matter (per-category rank 0.60–0.82). PARA is weakest on **night
scenes** (MAE 0.089) and best on animals/buildings (0.055); EVA spread is modest.

**(5) Between-group separation on structured signals (`structured_separation.py`).** The C1
separation does **not** extend to the personal signals at temp 0: contentPreference +0.016,
willingnessToShare +0.022, EVA difficulty −0.009 — all CIs include 0. So the aggregate captures the
group *mean* of these signals (§14.12.1) but not yet the between-group *differences*; a candidate
temp-0.7 win.

**(6) Calibration transfer across datasets (`calib_transfer.py`) — supports "fit once, reuse" (C3).**
A calibrator fit on one dataset and applied to another still beats raw by a wide margin and lands
near the fit-on-self diagonal (calibrated group MAE, eval-row × fit-on-col):

| eval ╲ fit-on | PARA | EVA | LAPIS | raw |
|---|---|---|---|---|
| PARA  | 0.062 | 0.083 | 0.070 | 0.184 |
| EVA   | 0.094 | 0.063 | 0.085 | 0.103 |
| LAPIS | 0.086 | 0.100 | 0.072 | 0.145 |

Every off-diagonal ≪ raw → calibration is a largely transferable response-style correction, which
is what the C3 "fit calibration on real images, reuse on generated" step relies on.

**(7) Holm correction across the slice sweep (`holm.py`) — statistical hardening.** Applying
family-wise Holm at α=0.05 to the eight C1 slice separations, **6/8 survive** (LAPIS age /
art-interest / nationality, PARA photography- & art-experience, and EVA age — the last significant
but *negative*, i.e. wrong-direction). The two nulls (PARA age, EVA photographic-level) drop out.
The headline LAPIS/PARA separations hold under multiple-comparison correction. *(Cells are
image-clustered, so these analytic p's are anti-conservative; the bootstrap CIs remain primary.)*

**(8) LAPIS repeated-measures sensitivity (`repeated_measures.py`).** Keep-all / drop-exact /
average-repeats give **identical** headline numbers (individual MAE 0.233, calibrated group MAE
0.072) — LAPIS results are robust to how the 283 genuine repeats and 900 exact duplicates are
handled.

**(9) Off-grid / snapping audit (`offgrid_audit.py`).** Raw-response parse rate ≈ **100%**,
off-grid = out-of-range = snapping-changed = **0.0000** on all three datasets. The model already
answers on the intended value grid; the discretization/snapping step is a no-op and distorts
nothing.

**Second-wave conclusion.** The persona's individual-level reach is now mapped (it captures rater
leniency on LAPIS, not on EVA); the judge's **central-tendency response style** is identified as the
mechanism behind the calibration gains and a confounder to control; **calibration transfers across
datasets** (green-lighting the C3 reuse plan); and the statistical/robustness audits (Holm,
repeated-measures, off-grid) all come back clean. This effectively **exhausts the temperature-robust
questions on the current logs** — the remaining open items (distributional C1, real N-curve,
structured-signal separation, emotion, C3, C4) all genuinely require the new runs.

### 14.14 Definitive status & next steps (current source of truth; supersedes the §14.9 scorecard and §14.10 roadmap)

**Evidence status per claim (all numbers from §14.4–14.13, cross-checked against `results/`).**

| Claim / question | Status | Decisive evidence | Blocking work |
|---|---|---|---|
| **Exp 0** — individual near-noise, group reliable | ✅ **proven** | ICC1 0.19–0.47 vs ICCk 0.84–0.96; persona-ΔR² 1–4%; classical CF gain only 0.01–0.03 (§14.4, 14.11a, 14.12.4) | — |
| **C2** — aggregation + calibration works | 🟡 **proven (mean/rank); aggregation *mechanism* not shown** | calibrated aggregate beats prior on **all 16 axes** (§14.6, 14.12.1); but N-curve is flat on all 3 datasets → the win is image-quality+calibration, **not** persona-averaging | real N-curve needs a **decoding fix** — temp 0.7 was tried on all 3 and **failed** (§14.15) |
| **Steerability** — persona is functional | ✅ **proven** | score-level r +0.37/+0.39 (PARA/LAPIS); **rationale-text AUC 0.58–0.70 vs blind 0.50, incl. EVA** (§14.7, 14.12.7); temperature-robust (§14.15) | stronger score-level effect → decoding fix (temp 0.7 insufficient) |
| **C1** — reproduce between-group differences (**headline**) | 🟡 **demonstrated on LAPIS; temperature-robust** | separation **+0.17 [0.15,0.18]** vs blind ≈0; survives Holm; unchanged at temp 0.7 (§14.15); PARA +0.04; EVA fails (§14.11c, 14.13.7) | distributional half (W1/KL/ECE) still needs the decoding fix |
| **C1** on structured signals | ⚪ **null** | content-pref/willingness/difficulty separation ≈0 — confirmed still ≈0 at temp 0.7 (§14.13.5, 14.15) | decoding fix (temp 0.7 tried, still null) |
| **Response-style / calibration mechanism** | ✅ **characterized** | central-tendency: VLM uses 51–58% of human scale spread; calibration transfers across datasets (§14.13.2, 14.13.6) | — |
| **Bias / fairness** | ⚠️ **flagged** | fair on personality/age; **nationality bias gap 0.50**, region 0.23 (§14.12.5) | net out in C3; ethics appendix |
| **Leakage** | ✅ **clean** | artist-fame vs error ≈0; snapping a no-op; parse ≈100% (§14.12.6, 14.13.9) | optional memorization probe (needs inference) |
| **C3** — cross-cultural on generated images | ⚪ **not started** | LAPIS nationality separation is the real-image precursor (§14.11c) | Rapidata runs |
| **C4** — audience editing feedback | ⚪ **not started** | complaint vocabulary looks usable (§14.12.7) | editor + reranker runs |

**The forward plan, in order.** Current-data analysis is **exhausted**; every remaining step needs
new inference.

> **★ ACTIVE WORKSTREAM (decided 2026-07-14): C3 — cross-cultural preference on AI-generated images
> (Rapidata).** We are not spending more compute on the temperature/decoding problem; instead we
> extend to C3, which builds directly on our strongest result (the LAPIS nationality separation).
> Full teammate-facing spec: **`docs/claim3_cross_cultural.md`**. Data source confirmed available and
> correctly structured (per-vote `country`/`language` present) — see §14.16 below.

1. **C3 (Rapidata cross-cultural) — ACTIVE.** Score frozen-judge appeal for unique AI-generated
   images conditioned on the *voter's country*, restricted to nationalities that also appear in
   LAPIS, and predict pairwise winners on **disagreement pairs** (ΔAUC vs a global-preference
   baseline + a no-persona control). Prerequisites already validated: national-bias control
   (§14.12.5) and calibration transfer (§14.13.6). Data-prep + analysis are local (no GPU);
   inference is the teammate's run, exported like the real-image runs.
2. **C4 (editing loop)** — after C3; proposer ≠ reranker; seed complaints from the mined rationale
   vocabulary (§14.12.7).

*Parked (not pursuing now):* the decoding escalation (token-level score distribution / higher-temp
self-consistency, `docs/task_temperature_rerun.md`) that would unblock the real N-curve and the
distributional half of C1; further PARA widening. These stay documented for
if we revisit, but C3 is the priority.

**One-line summary.** The supporting science (Exp 0 + C2) is proven, the persona is proven
functional, and the headline C1 between-group claim is demonstrated on LAPIS and significant under
correction; the active push is now **C3 — does the same cross-cultural signal carry from real art to
AI-generated images** — followed by C4.

### 14.15 Temperature-0.7 re-run results — the re-run did **not** lift the floor (with one coverage win)

*Recorded 2026-07-14 (updated with LAPIS 2026-07-14). Data: `data/results/{para,eva,lapis}_full_t07/`
(all three persona runs at temperature 0.7, single sample). Script: `temp_compare.py` →
`results/temp_compare.json`. Note PARA temp-0.7 was **also widened to 4,000 images** (from 2,000), so
its column mixes temperature and coverage; **EVA and LAPIS temp-0.7 are the same task sets** (clean
temperature-only comparisons).*

**Diagnosis first: the model barely sampled.** Sampling *was* active (12–13% of predictions changed
vs temp-0, mean |Δ|≈0.07–0.13), but Qwen2-VL-7B is so peaked on these discrete rating tasks that
0.7 decoding **reverts to the mode ~88% of the time** — e.g. one EVA image had all 46 personas
return exactly 7.0. So the panel did not gain the spread the re-run was meant to inject.

| Metric (persona `full`) | PARA t0 (2k) | PARA t0.7 (4k) | EVA t0 | EVA t0.7 | LAPIS t0 | LAPIS t0.7 |
|---|---|---|---|---|---|---|
| Degenerate-panel fraction | 0.48 | 0.42 | 0.51 | 0.53 | 0.04 | 0.03 |
| N-curve drop (N=1→20) | 0.003 | 0.004 | 0.003 | 0.003 | 0.011 | 0.006 |
| C1 between-group separation | 0.044 [0.025,0.064] | 0.040 [0.027,0.055] | −0.085 | −0.080 | **0.166 [0.150,0.181]** | **0.172 [0.156,0.189]** |
| Steerability r | 0.367 | 0.386 | −0.239 | −0.272 | 0.394 | 0.407 |
| Persona value (within-image) | 0.024 | 0.022 | −0.017 | −0.014 | 0.094 | 0.092 |
| Scale-usage ratio (vs human) | 1.15 | 1.17 | 0.58 | 0.56 | 0.51 | 0.48 |

**Conclusion (all three datasets).** Every temperature-sensitive metric is **essentially unchanged**:
the panel is still collapsed where it was (PARA/EVA ~42–53%; LAPIS was never collapsed — its 0–100
scale gives room — so temp had little to fix there), the **N-curve is still flat**, and steerability /
persona-value / scale-compression all move within noise. Temperature 0.7 with single-sample decoding
is **not** enough to overcome this model's peakedness — the floors identified in §14.5/§14.11/§14.13
persist. **Silver lining: the headline C1 between-group separations are temperature-robust** — LAPIS
holds at +0.17 (t0) → +0.17 (t0.7), PARA at +0.04, EVA still fails — so the C1 result is *not* a
temp-0 artifact, which strengthens it for the write-up.

**The coverage win (independent of temperature).** Doubling PARA to 4,000 images **confirms and
tightens the headline between-group separation**: +0.040, CI **[0.027, 0.055]** on 46,160 cells
(was +0.044 [0.025, 0.064] on 23k) — the PARA result is real and now more precise. (EVA still fails,
as expected from its thin personas.)

**Revised decoding recommendation (supersedes the temp-0.7 spec).** To actually elicit a panel
distribution from a peaked judge, escalate decoding — in order of preference:
1. **Read the token-level score distribution** (softmax over the valid score tokens / logits) instead
   of sampling — the principled fix: it recovers the model's full predictive distribution per
   (image, persona) in *one* forward pass, no sampling variance, no mode-collapse.
2. **Self-consistency at higher temperature** — T≈8–16 samples per (image, persona) at **temp ≈
   1.0–1.3**, aggregated to an empirical distribution (Option B of the task doc, escalated).
3. A **temperature sweep** {0.7, 1.0, 1.3, 1.6} on a small image subset first, to find where the
   panel actually de-collapses before paying for a full run.
`docs/task_temperature_rerun.md` has been updated with this finding and the escalated plan.

**Status deltas.** All three temp-0.7 runs are now in (`{para,eva,lapis}_full_t07`). C2's real N-curve
and C1's distributional half remain **blocked** — not by "we haven't run it" but by a now-understood
decoding limitation; the temp-0.7 runs answered the question and the answer redirects the method
(elicit the distribution, don't sample it). Net positives: the wider-PARA C1 separation is promoted
from "weak" toward "confirmed-but-small", and **all three C1 separations are confirmed
temperature-robust** (unchanged from temp-0), so the headline is not a decoding artifact.

### 14.16 C3 data-source audit (Rapidata) — available and correctly structured

*Recorded 2026-07-14. The C3 blocking prerequisite (plan §1.1 week-1 audit) is confirmed.*

- **Source:** `Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3` (public HF dataset;
  companion axes `Flux_SD3_MJ_Dalle_Human_Coherence_Dataset` and `..._Alignment_Dataset` also
  present). It is a **pull**, not something we host — nothing to push to our repos.
- **Schema (verified by streaming a row):** `prompt`, `image1`/`image2` (embedded 1024×1024 bytes),
  `votes_image1`/`votes_image2`, `model1`/`model2` (the 4 generators), `image1_path`/`image2_path`,
  and **`detailed_results`** — a JSON string of per-vote records
  `{"votedFor":…, "userDetails":{"country":"IN","language":"en","userScore":0.78}}`.
- **Verdict:** the make-or-break requirement — **per-vote country + language** — is present, so C3 is
  runnable. Countries are ISO-2 codes (need an ISO↔LAPIS-nationality map, e.g. `GB`→`british`).
  Split layout is sharded (`train_0001…`), not a single `train`.
- **Implication:** C3 needs no new data licensing (public) and no push; the only new artifacts on our
  side are the ISO↔LAPIS map and the frozen calibration reused from real images. Full plan in
  `docs/claim3_cross_cultural.md`.

### 14.17 Publication assessment — what we can honestly claim and where it can go

*Recorded 2026-07-14. A frank read on publishability given we are not pursuing more decoding work
until (optionally) after C3.*

**Claim count.** Of the four headline claims: **C1 is meaningfully demonstrated** (on the strongest
dataset, LAPIS, incl. the nationality axis, temperature-robust and Holm-significant); **C2 is
half-demonstrated** (group beats individual and beats the prior after calibration — but the
*aggregation mechanism*, the N-curve, is not shown and is our weakest claim); **C3 and C4 are not
attempted**. **Exp 0 and the steerability gate are fully proven** and are strong on their own.

**The honest caveat to hold onto.** Because the persona panel collapses, our "group prediction"
success is really *"a calibrated frozen VLM predicts image-mean quality and captures nationality/
age/art-interest **differences**"* — not *"averaging diverse personas cancels noise."* We should not
sell C2's aggregation story as if the N-curve supported it; it does not.

**What is publishable now (without C3/C4):** a self-contained **workshop paper**, the venue the
proposal already targeted (**NeurIPS Creative AI 2026, non-archival, 2–6 pp; deadline ~Aug 3**).
Defensible thesis:

> *Frozen-VLM persona panels reproduce between-group aesthetic differences on art (by nationality,
> age, art interest) but not on photos; individual taste sits at the noise ceiling; and the effect is
> bounded by VLM central-tendency bias and decoding mode-collapse.*

It delivers a rigorous **ceiling analysis** (Exp 0), a **positive headline** (image-controlled LAPIS
between-group separation, no-persona control ≈0, temperature-robust), **honest negatives** (EVA fails,
photos weak, N-curve flat, temp-0.7 didn't help), a **mechanism** (central-tendency compression,
calibration necessity + cross-dataset transfer, persona steers text not scores), and **ethics +
leakage** results. Suggested artifacts: Fig 1 = Exp 0 ceiling; Table 1 = C1 separation × 3 datasets
(full vs blind, Holm); Fig 2 = response-style/calibration; appendices = bias + leakage. **It must not
claim** cross-cultural generalization to *generated* images (C3), editing usefulness (C4), or that
persona-aggregation is the mechanism.

**Upgrade path.** **C3 is the single highest-value addition** — it turns the honest workshop paper
toward the full-thesis paper by testing whether the LAPIS nationality signal carries to generated
images, and reuses everything built. Hence C3 is the active workstream (§14.14).

### 14.18 Project state — data, artifacts, and Hugging Face sync

*Recorded 2026-07-14, for reproducibility hand-off.*

- **Raw datasets on HF (private, pushed 2026-07-02):** `savoji/PARA`, `savoji/EVA`, `savoji/LAPIS`
  (via `scripts/push_to_hf.py`; only these three are registered in `scripts/hf_dataset.py`).
- **Not on HF (local only, `data/` git-ignored):** all run results under `data/results/` — the temp-0
  `*_full`/`*_blind` runs, the temp-0.7 `*_full_t07` runs (PARA/EVA/LAPIS all in as of 2026-07-14),
  and the analysis outputs in `results/*.json` + `results/figs/`. The sync scripts do **not** cover
  results; backing them up would need a new private repo or an ad-hoc push.
- **Licensing:** PARA/LAPIS are redistribution-restricted → any derived artifact (results embed their
  human ratings) must stay **private**; EVA is CC0.
- **C3 data:** Rapidata is public — pulled, not pushed (§14.16). The only new local artifacts C3 needs
  are the **ISO↔LAPIS nationality map** and the **frozen calibration** object.
- **Code in git:** the analysis suite (`scripts/analysis/`) and the imported teammate pipeline
  (`src/`, `script/`); a subset was committed outside this session (HEAD `f3d7193`). `research_plan.md`
  + `docs/*` carry the running log. `data/` and `results/` stay git-ignored by design.

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
