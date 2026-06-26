# Research Plan — Synthetic Audiences for Creative AI

Step-by-step plan to implement, run experiments, and produce results. Everything is Hugging Face–based and **serverless-first via Hugging Face Inference Providers** for VLM/editor calls; no backbone is trained, fine-tuned, or self-hosted for the main result. Local code handles data processing, caching, calibration, metrics, and classical baselines. Maps directly to the claims in `PROPOSAL.md`: **Exp 0** (ceiling), **C1** (group reaction prediction — headline), **C2** (why aggregation works: aggregate-vs-individual gap + N-curve), **C3** (generalization to generated, cross-cultural), **C4** (editing feedback).

**Guiding rules**
1. **Frozen/serverless only**: all VLM/editor/reward calls use off-the-shelf provider-hosted models through HF Inference Providers where available. The only "fitting" allowed is (a) post-hoc output calibration (maps scores, no weights) and (b) classical non-VLM baselines (regressor/CF). Never update, download, or self-host a judge backbone for the MPR.
2. **MPR first**: Exp 0 + C1 (group prediction) + C2 (aggregation gap + N-curve) on one pinned serverless HF Inference Providers VLM is the minimum publishable result.
3. **Persona is text, image is the only visual input.**
4. **Pre-register** primary statistical endpoints + the population-mean baseline before looking at test results.
5. **No proposer = reranker** in C4. Bootstrap CIs on every headline number.

---

## 0. Repository & environment

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
**Hardware target:** no local GPU is required for the core MPR. **Everything is inference** — judge, editor, and reward models are frozen and called through provider APIs where available. The cost is API volume, image transfer, and rate limits (see Appendix E).

> **Serverless pins to verify in week 1:** exact model IDs callable through HF Inference Providers, provider suffix/policy (`:fastest`, `:cheapest`, or explicit provider), image input format (`image_url` vs. base64/data URL), JSON/schema support, rate limits, billing account, content filters, and whether FLUX/EditReward routes are available serverlessly. A model being on the HF Hub is not enough.

---

## 1. Data: acquisition, schema, splits

Goal: one **unified per-(image, rater) parquet** plus a **per-(pair, slice) parquet** for generated data. Normalize every rating to `score_norm ∈ [0,1]`.

### 1.1 Datasets and how to get them

| Dataset | Source | Used for | Action |
|---|---|---|---|
| **PARA** | project page / request | C1, C2, Exp 0 | request wk1; expect CSV of (image, user, score, attributes) |
| **LAPIS** | [GitHub](https://github.com/Anne-SofieMaerten/LAPIS) request | C1, C2, Exp 0, nationality for C3 | request wk1; **audit nationality coverage** |
| **ArtEmis** | `datasets.load_dataset("youssef101/artelingo")` (English subset) | warm-up language | filter `lang=="en"` |
| **RPCD** | project page | warm-up critique language | optional; skip if blocked |
| **Rapidata** | `load_dataset("Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3")` (+ companion sets) | C3 | parse `detailed_results` for per-vote `country`,`language` |
| **RichHF-18K** *(optional)* | HF dataset / extract via Pick-a-Pic | extra fine-grained generated-image sanity check / exemplar pool | not required |
| **EditReward-Data / -Bench** | `load_dataset("TIGER-Lab/EditReward-Data")`, `("TIGER-Lab/EditReward-Bench")` | C4 | preference pairs + bench |

**Week-1 license/serverless availability audit (blocking):** confirm (a) PARA/LAPIS redistribution terms, (b) Rapidata license + that image **bytes** or provider-accessible URLs are present, (c) LAPIS nationality list, (d) whether dataset terms permit sending images/persona metadata to hosted inference providers, and (e) which candidate VLM/editor/reranker models are actually callable through HF Inference Providers.

### 1.2 Unified rating schema (`processed/ratings.parquet`)
```
image_id, image_path, dataset {para,lapis},
user_id, score_norm[0..1],
# structured (PARA only; null for LAPIS):
emotion, content_pref, willingness_share, difficulty,
# persona attributes (per user):
age_bucket, gender, education, art_exp, photo_exp,
big5_O, big5_C, big5_E, big5_A, big5_N, nationality
```
Normalize: PARA score (1–5) → `(s-1)/4`; LAPIS (0–100) → `s/100`.

### 1.3 Splits (`splits/`) — the C1 design hinges on this
The "reference pool" = the ratings used to fit calibration, draw few-shot exemplars, and fit the classical baselines (regressor/CF). The frozen VLM never sees any of it as gradients.
- **cold_start**: hold out a disjoint set of **users** (e.g., 20%); none of their ratings enter the reference pool. *Primary C1 regime — the VLM predicts for a viewer it has no data on, from the persona card alone.*
- **warm_start**: same users in pool/test, hold out (user, image) pairs. *CF baseline lives here only.*
- **image-disjoint** and, for LAPIS/WikiArt, **artist-disjoint** test folds for leakage control.
- Save split indices as JSON; fix a global `seed=0`.

### 1.4 Generated-image parquet (`processed/rapidata_pairs.parquet`)
```
pair_id, prompt, image_a_path, image_b_path, model_a, model_b,
slice_key (country|language), votes_a, votes_b, n_votes
```
Aggregate per (pair, slice): `winrate_a = votes_a/(votes_a+votes_b)`. Keep slices with `n_votes ≥ 30`; restrict to **top-N slices by volume** and to **nationalities present in LAPIS**.

---

## 2. Persona card (the conditioning input)

Deterministic template from attributes; missing fields omitted (never invented).
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
For Rapidata (C3) only `country`/`language` are available → card uses only those fields. **Critical:** the card builder must produce the *same field set* in the reference pool and at C3 test for any field you want to claim generalization on — so the cross-cultural claim only covers `nationality/country`.

---

## 3. Models (HF Inference Providers wrappers)

`src/models/judge.py` — one class, swappable serverless backbone.
```python
from huggingface_hub import InferenceClient

class Judge:
    def __init__(self, model_id, provider="auto", bill_to=None):
        # model_id examples after week-1 audit:
        # "zai-org/GLM-4.5V", "CohereLabs/aya-vision-32b:cohere",
        # or any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM.
        self.model_id = model_id
        self.provider = provider
        self.client = InferenceClient(provider=provider, bill_to=bill_to)

    def score(self, image, persona_card) -> dict:
        # user content = [image_url or base64 data URL, persona_card, output-format]
        # request response_format=json_schema when supported
        # parse JSON {score, emotion, willingness, difficulty, rationale}
```
- **Judge:** unified default = one **HF Inference Providers conversational VLM** pinned after the week-1 availability audit. Seed candidates: `zai-org/GLM-4.5V`, `CohereLabs/aya-vision-32b:cohere`, plus any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM. The old open-weight shortlist is not part of the MPR unless those exact models become serverless-callable.
- **Serverless audit log:** for every candidate, record model ID, provider, task support, image input support, JSON/schema support, max context/output, pricing/rate limits, content filters, and smoke-test parse rate. Save rejected candidates too.
- **Output format:** request JSON/schema mode when supported; also force JSON via the prompt; parse with a tolerant parser; on parse-fail, retry once then mark NaN.
- **Reward wrappers** (`src/models/rewards.py`): serverless-first. Use EditReward only if a provider route exists; otherwise evaluate C4 with EditReward-Bench labels and serverless scalar/evaluator baselines while preserving proposer != reranker.
- **Editor** (`src/models/editor.py`): serverless-first HF provider calls. Audit `black-forest-labs/FLUX.1-Kontext-dev` via provider route; SD-3.5-Medium or another provider-backed editor is fallback.

---

## 4. Exp 0 — Ceiling analysis (run FIRST)

Bounds how predictable *individual* taste is, and **motivates aggregation**: it quantifies how much idiosyncratic variance the group average is expected to cancel (the basis of C2). Frames every individual-level number.

`src/eval/exp0_ceiling.py`
1. **Inter-rater reliability:** ICC(2,k) and mean pairwise Spearman across raters per image (PARA, LAPIS separately).
2. **Variance decomposition** via linear mixed model (`statsmodels`):
   `score_norm ~ 1 + (1|image) + (1|user) + persona_fixed_effects`
   Report variance fractions: image, user, persona-explained (ΔR² of persona fixed effects), residual noise.
3. **Predictability ceiling:** max achievable individual-level R² ≈ `1 − noise_fraction`; the persona-attributable ceiling = ΔR²(persona).
4. **Output:** `results/exp0.json` + a variance bar chart. **This number is reported next to every individual-level result and anchors C2's aggregate-vs-individual gap.**

**Decision gate:** a low persona ΔR² is *expected and fine* — it is the motivation for aggregating. Only if the **group** signal also vanishes (C1 fails) do we pivot toward "frozen VLMs can't simulate group taste" as the finding.

---

## 5. Stage 1 — AudienceJudge (frozen, no training)

### 5.1 Prediction recipe (`src/predict/`)
The judge is a **frozen serverless VLM queried with a prompt** through HF Inference Providers. Output = JSON `{score, emotion, willingness, difficulty, rationale}` (Appendix A). The only tunable parts are prompt + post-processing — never weights:

1. **Zero-shot** persona-conditioned prompting (default per-persona judge mode; the headline method is panel-and-aggregate over personas — §6).
2. **Few-shot in-context** (optional): prepend k=2–8 exemplars. Either *generic* (fixed diverse set) or *retrieved* — nearest neighbours in CLIP/DINOv2 image space (and/or persona space) from the **reference pool**, via `faiss`. Exemplars carry their (persona, reaction) so the model copies the mapping in-context.
3. **Self-consistency** (optional): sample T outputs (temperature>0), aggregate (median score, majority emotion).
4. **Post-hoc calibration** (`src/predict/calibrate.py`): fit isotonic regression (or temperature) mapping raw→calibrated score on a **held-out real-image calibration split**. This corrects scale/bias for the ECE/distributional metrics and **changes no model weights**. Fit once on real images; reuse unchanged for C3 (its generalization is part of the C3 claim).

Provider mechanics:
- Use `AsyncInferenceClient`/`httpx` workers with bounded concurrency, exponential backoff, and explicit retry handling for 429/5xx/provider failover.
- Store raw request/response metadata: model ID, provider, policy suffix, prompt version, decoding params, response format, token/image usage when returned, latency, and parse status.
- Images must be provider-accessible: prefer local byte-to-data-URL only if the provider supports it; otherwise use release-safe signed/public URLs generated during preprocessing.

### 5.2 Steerability gate (`src/eval/steerability.py`) — run before trusting any number
- Build **data-driven expected directions**: from the reference pool, compute each attribute's empirical mean-score effect (e.g., high-art-familiarity → +Δ on abstract art).
- Permute/ablate fields in the persona card given to the **frozen** judge; measure prediction shift; correlate observed shifts with the empirical directions.
- **Metric:** steerability = corr(predicted Δ, empirical Δ); also % attributes with correct sign.
- **Gate:** if ≈0, the frozen model ignores the persona → try few-shot/stronger prompt/larger backbone; if still null, **pivot the paper to the ceiling/“frozen VLMs can’t personalize” finding** (still publishable).

---

## 6. Synthetic audience (`src/audience/`) — the core method (C1)
- `sample_personas(target_distribution, N)` → N cards drawn to match a **target group** (a slice spec) or the dataset marginal. The target distribution defines *which group* you simulate.
- `predict_group(image, personas)` → run frozen judge per persona → empirical group distribution.
- `aggregate(distribution)` → mean, dispersion, top complaints (from rationales), between-group disagreement summary.
- **N sweep** (`sweep_N`): recompute the group distribution at N = 1, 2, 5, 10, 20, 50 → feeds the C2 fidelity curve.
- Cache judge outputs keyed by `(image_id, persona_hash, model_id, provider, prompt_version, decoding_params)` so the panel/N-sweep/slices are all assembled from one query per (image, persona, model, prompt).

---

## 7. Stage 2 — Audience-guided editing (`src/eval/c4_editing.py`)
1. Input image → sample audience → aggregate complaints + between-group disagreements.
2. **Judge (proposer)** distills ONE edit instruction (constrained short text).
3. **Serverless editor** (FLUX.1-Kontext provider route when available; otherwise a provider-backed editor fallback) → K candidates (K=4).
4. **Rerank with held-out signals only**: EditReward if serverless-accessible; otherwise EditReward-Bench labels / serverless scalar evaluators. *Proposer never reranks.*
5. Conditions to compare: `{no-feedback, generic-VLM-critique, EditReward-only, audience-aggregate, audience-targeted}`.

---

## 8. Experiments & metrics

`src/metrics/` implements all; every headline metric returns a **bootstrap 95% CI** (resample over raters and images, 1000 draws).

### 8.1 C1 — group prediction (headline)
Predicted **group reaction distribution** (aggregate of N personas) vs. the observed group (~25 raters/image), pooled per slice.
| Metric | Baselines | Pass criterion |
|---|---|---|
| Wasserstein-1, KL, ECE of predicted vs observed **group** distribution (pooled per slice) | **no-persona aggregate**, **population-mean prior** | lower distributional error than both (CI excludes 0) |
| **Between-group separation**: corr(predicted slice-to-slice gaps, observed gaps) | no-persona (≈0 separation) | predicted gaps track observed gaps |
- Slices = age bucket × art familiarity × (nationality where available). Min slice size enforced; reliability diagram per slice.

### 8.2 C2 — why aggregation works
- **Aggregate-vs-individual gap:** same model, report per-rater individual error (Spearman ρ, MAE; vs. Exp-0 ceiling) **alongside** the C1 group error → show individual ≫ group error.
- **N-personas fidelity curve:** group distributional error vs. panel size N ∈ {1,2,5,10,20,50}; expect monotone decrease + saturation. Fit/report the saturating-N.
- **Warm-start reference (individual only):** collaborative filtering (`surprise`/MF) on seen users — reported as the individual lower-bound reference, not a group baseline.

### 8.3 C3 — generalization to generated (Rapidata)
- Convert judge scores to pairwise: `p̂(a≻b | slice) = σ(β·(ŝ_a − ŝ_b))` (fit β on a val split).
- **Define disagreement pairs:** pairs where slice win-rates differ across the top slices by `|Δwinrate| > 0.2` with `n_votes ≥ 30` each.
- **Primary metric:** pairwise accuracy **on disagreement pairs**, and **ΔAUC = AUC(slice-conditioned) − AUC(global-preference model)**. Report on the **LAPIS-nationality subset**.
- **Controls:** (i) **no-persona prompt** (separates persona signal from default judgment); (ii) **global-preference** baseline (separates cross-cultural signal from universal quality).
- Report the **real→generated gap** vs C1.

### 8.4 C4 — editing usefulness
- **EditReward-Bench:** ranking agreement (accuracy / Kendall-τ) of each condition.
- **C4-core (slicing-independent):** win-rate of audience-aggregate vs single-judge vs no-feedback (EditReward/serverless held-out evaluator if available, EditReward-Bench labels, + optional human).
- **C4-targeted:** within-group win-rate; **contingent on C3 showing disagreement-pair signal**. If between-group variance is null, report C4-core only.
- **Secondary:** edit-drift (DINO/CLIP-I identity similarity to source).

### 8.5 Leakage diagnostics
- Memorization probe: prompt judge to reproduce ArtEmis caption/title from image; report hit-rate.
- Compare seen vs. artist-disjoint performance; report the gap.

---

## 9. Statistical analysis plan (`preregistration.md`)
- **Primary statistical endpoints (fix before test):** (1) **C1 group-distribution Wasserstein vs no-persona aggregate**; (2) C1 between-group separation; (3) C2 N-curve saturating-N + aggregate-vs-individual gap; (4) C3 ΔAUC on disagreement pairs; (5) C4-core win-rate vs single-judge.
- **CIs:** 1000-sample bootstrap, clustered by rater and by image.
- **Multiple comparisons:** Holm correction across the subgroup/slice sweep (secondary analyses).
- **Significance:** report effect size + CI, not just p; a result "counts" only if its CI excludes the null *and* it sits within the Exp-0 ceiling.

---

## 10. Ablations (all frozen — these are the method's only knobs)
1. Persona-card field ablation (drop each field group; ΔWasserstein).
2. Zero-shot vs few-shot (generic vs retrieved exemplars; k sweep).
3. Self-consistency (T samples) vs single sample.
4. With vs. without post-hoc calibration.
5. Aggregation vs single judge, and **persona-sampling scheme** (how the panel is drawn) — core to C1.
6. Serverless judge/provider family: primary pinned VLM vs 2-4 additional HF Inference Providers VLMs that pass the week-1 audit. Desired open-weight families (Gemma/Qwen/Kimi/Phi/Nemotron) are included only if provider-callable.
7. Reranker source (EditReward vs scalar) for C4.
8. *(Optional, non-validated)* peer-revision audience probe.

---

## 11. Reproducibility & artifacts
- Log every run with `seed`, config hash, model id, prompt version, dataset version.
- Save `results/*.json` (metrics + CIs) and auto-generate tables/figures.
- Release-safe outputs only (respect PARA/LAPIS/Rapidata licenses); release code + (if permitted) the optional benchmark at camera-ready.

---

## 12. Week-by-week build order (with go/no-go gates)

| Wk | Build | Gate |
|---|---|---|
| 1 | env; license/nationality audit; HF Inference Providers model audit; download HF data; unify parquet; build splits; frozen serverless Judge wrapper | data loads, splits valid, primary provider/model pinned |
| 2 | **Exp 0 ceiling**; persona-card builder; prompt design; **steerability gate** | persona ΔR²>0 and steerability>0 (else pivot) |
| 3 | post-hoc calibration; **C1 group prediction + C2 aggregation gap/N-curve + baselines** → **MPR locks** | C1 beats no-persona aggregate; N-curve improves with N |
| 4 | **C3 on Rapidata** (zero-shot; disagreement ΔAUC + no-persona/global controls); leakage | C3 ΔAUC CI excludes 0 (else report null honestly) |
| 5 | Editing loop; **C4-core on EditReward-Bench**; serverless cross-model/provider panel on C1+steerability; few-shot/self-consistency | C4-core win vs single-judge |
| 6 | Ablations; bias/fairness diagnostics; figures; writing (2–6 pp) | statistical endpoints reported with CIs |

**Stretch (only if MPR is solid):** C4-targeted, optional human study, second editor (SD-3.5).

---

## 13. First commands (smoke test, day 1)
```bash
export HF_JUDGE_MODEL="CohereLabs/aya-vision-32b:cohere"  # replace after week-1 audit
python -m src.data.download --datasets artelingo richhf rapidata editreward
python -m src.data.build_unified         # → processed/ratings.parquet
python -m src.data.make_splits --seed 0  # → splits/
python -m src.models.judge --backend hf_provider --model "$HF_JUDGE_MODEL" --provider auto --smoke
python -m src.eval.exp0_ceiling          # → results/exp0.json
```

---

## Appendix A — Judge prompt & output schema

Same template for every backbone; only the provider request wrapper differs.

**System:**
```
You are simulating one viewer's reaction to an image. You are given the viewer's
profile and one image. Predict THIS viewer's reaction, not a general opinion.
Respond ONLY with JSON matching the schema. Scores are 0.0–1.0.
```
**User:** `[IMAGE]` + persona card (Appendix-A card from §2) + the schema:
```json
{"score": 0.0, "emotion": "awe|amusement|sadness|fear|disgust|anger|contentment|excitement|other",
 "willingness_share": 0.0, "difficulty": 0.0, "rationale": "<=25 words"}
```
- **Decoding:** `temperature=0`, `max_tokens=128`. Prefer provider JSON/schema mode if supported; else tolerant parse + 1 retry, then NaN.
- **For art (LAPIS):** only `score` + `rationale` are scored (no structured fields).
- **Edit-instruction prompt (C4 proposer):** input = aggregated complaints → output one imperative edit ≤15 words, no rationale.

---

## Appendix B — Baselines (exact implementations)

| Baseline | Implements | How |
|---|---|---|
| **No-persona aggregate (C1)** | "does persona matter for the group?" | run judge with empty profile, aggregate N copies → group baseline |
| **Population mean (C1)** | group floor | per-image mean of reference-pool raters; constant across groups |
| **Single-judge (C2)** | aggregation value | one persona instead of aggregated N (also the N=1 point of the curve) |
| **Features+metadata regressor (C2 ref)** | non-VLM personalization | image features (frozen CLIP/DINOv2 from HF) ⊕ one-hot persona → GradientBoosting/MLP (`sklearn`) → score |
| **Collaborative filtering (C2 ref)** | warm-start individual ref | matrix factorization (`surprise` SVD/ALS) on user×image; **warm-start only** (cold-start cannot predict) |
| **Global preference (C3)** | "is it just universal quality?" | mean win-rate ignoring slice |
| **No-persona prompt (C3)** | "does the persona matter?" | same frozen judge, empty profile |
| **EditReward-only (C4)** | reward-guided editing | pick edit maximizing EditReward if serverless-accessible; otherwise use a serverless held-out evaluator/bench-label variant, no audience critique |

*Note: the features+metadata regressor and CF are **classical non-VLM baselines** (sklearn/`surprise`) fit on frozen embeddings or the rating matrix — no backbone weights are touched, consistent with the frozen-only rule. They exist to show persona-prompting beats simple personalization, not as part of the method.*

---

## Appendix C — Metric definitions (`src/metrics/`)

- **Spearman ρ** (per rater, then mean): `scipy.stats.spearmanr`.
- **MAE**: mean |ŝ − s| on `score_norm`.
- **Wasserstein-1**: `scipy.stats.wasserstein_distance(pred_samples, obs_samples)` per slice, averaged.
- **KL(obs‖pred)**: histogram both into `B=10` shared bins on [0,1], add ε=1e-6, `sum(p·log(p/q))`.
- **ECE**: `M=10` confidence bins; `Σ (|B_m|/N)·|acc(B_m) − conf(B_m)|` (for the pairwise/binary predictions).
- **ICC(2,k)**: two-way random-effects via `pingouin.intraclass_corr` (or statsmodels variance components).
- **ΔAUC (C3)**: `roc_auc_score(label_a_wins, p̂_slice) − roc_auc_score(label, p̂_global)` on disagreement pairs.
- **Between-group separation (C1)**: `pearsonr(predicted slice-pair gaps, observed slice-pair gaps)`.
- **Bootstrap CI**: resample units (images/raters for C1–C2, image-pairs for C3) with replacement ×1000 → 2.5/97.5 percentiles. Cluster by rater *and* image where both apply.

---

## Appendix D — Example config (`configs/predict_persona.yaml`)
```yaml
backend: hf_provider
model: ${HF_JUDGE_MODEL}       # frozen serverless VLM, pinned after week-1 audit
provider: auto                 # or explicit provider, e.g. cohere
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

All model calls are inference-only; no training cost and no local GPU requirement for the MPR.

| Job | Cost driver | Estimate/control |
|---|---|---|
| Exp 0 | CPU mixed model | minutes locally; no provider calls |
| Audience inference (C1/C2) | **N personas × images × provider models** | dominant serverless bill; cache by `(image_id, persona_hash, model_id, provider, prompt_version)`; sweep N up to 50 from cached calls |
| C3 on Rapidata | ~tens of k images, 1 query per slice/persona design | reuse cached per-image/persona scores across pairs/slices; cap slices before querying |
| Editing (C4) | K candidates × inputs through provider editor | limit inputs to ~200; cache generated candidates and evaluation calls |

Rule: **query each image once per `(persona, model, provider, prompt)`**, then assemble panels/slices/pairs from the cache — never re-run the VLM per pair.

---

## Appendix F — Deliverables → claim → paper artifact

| # | Output | Claim | Paper artifact |
|---|---|---|---|
| Exp 0 | variance fractions + ceiling | motivates aggregation | Fig 1 (variance bars) |
| C1 | **group-distribution error vs no-persona/population + between-group separation** | C1 (headline) | Table 1 + Fig 2 (reliability per slice) |
| C2 | **aggregate-vs-individual gap + N-personas curve** | C2 (why it works) | Fig 3 (N-curve) + gap row |
| Steerability | corr(pred Δ, empirical Δ) | validity gate | Table 1 footnote |
| C3 | ΔAUC on disagreement pairs + no-persona/global controls | C3 | Table 2 |
| C4 | win-rates (core + targeted) + edit-drift | C4 | Table 3 + qualitative figure |
| Ablations | field / few-shot / self-consistency / calibration / aggregation / sampling / backbone | robustness | Table 4 |
| Bias | subgroup calibration error | ethics | Appendix |

---

## Appendix G — Optional human study protocol (only if pursued)

- **Platform:** Prolific / in-house; consent + de-identified persona intake (PARA/LAPIS fields + Big-5 short form, e.g. BFI-10).
- **Part A (psychographic groups, extends C3):** ~300 generated images (FLUX.1-dev/SD-3.5), ≥10 raters/image; endpoint = pooled subgroup distributional error.
- **Part B (group-targeted, extends C4):** forced-choice over the edit conditions; endpoint = within-group win-rate.
- **Power:** size for the *pooled subgroup* endpoint (primary); pre-register sample size from a pilot variance estimate; individual-level is exploratory.
- **IRB:** single protocol, submit week 1; if it slips, ship on public data (paper does not depend on this).
- **Ethics:** frame outputs as dataset-sampled distributions, not group essences; report over/under-statement of group gaps.

---

*Plan complete. Start with §13 smoke test, then follow §12 week-by-week, honoring the go/no-go gates.*
