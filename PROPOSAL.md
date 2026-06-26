# Synthetic Audiences for Creative AI: Persona-Conditioned Multimodal Judges that Predict Human Reactions and Guide Image Editing

> **Thesis.** Calling a frozen VLM through **Hugging Face Inference Providers** as a **panel of many personas** and aggregating their reactions can simulate the **collective reaction of a group** to art and AI-generated images — predicting the *group's* reaction distribution better than non-personalized baselines — and that simulated audience is a *non-circular* feedback signal for image editing.
>
> **Novelty in one line.** Not another personalized-aesthetics model: we use **off-the-shelf frozen VLMs via serverless API calls (no fine-tuning or self-hosted judge weights)** to predict the **group/collective** reaction (not one user), show that **aggregation makes prediction tractable even though individual taste is near-unpredictable**, and use the simulated audience as interpretable creative feedback.

---

## 1. Motivation

"Is this image good?" is not a scalar — it is the *collective* reaction of a group, a distribution across viewers who differ in culture, training, and taste. Two questions follow: (i) can we **simulate the group's reaction** to a given (real or generated) image, and (ii) is that simulated audience a *useful* feedback signal for creative editing? The unit of interest is the **group**, not the individual — and a key finding is *why* that matters: individual taste is near-unpredictable, yet the group aggregate is predictable. This fits the NeurIPS Creative AI "Agency" theme directly — when a system can predict a group's aesthetic preference, creative agency shifts among audience, model, and creator. It is distinct from text-only "silicon sampling": our audience reacts to *visual* stimuli.

**Positioning.** *PAMELA* (2026) personalizes a scalar reward to a single user and optimizes prompts; we model a population distribution, keep the generator frozen, and feed back natural-language critique. *AesBiasBench* (2025) supplies our subgroup-calibration lens. Work on *individual differences in computational aesthetics* (2025) motivates our explicit ceiling analysis.

---

## 2. Claims

- **C1 — Group reaction prediction (headline).** Running the frozen judge as a **panel of N personas sampled to match a target group**, then aggregating, predicts the group's **collective reaction distribution** to an image better than (a) a **no-persona** judge (same prompt, empty profile) and (b) the **population-mean prior**. Validated on real images where each image has a real group of ~25 raters (PARA/LAPIS) via distributional metrics (Wasserstein/KL/ECE), **pooled per group/slice** (not per-image-per-slice). The decisive evidence is capturing **between-group differences**, not just the global average. Scope: **score** on photos (PARA) + art (LAPIS); **structured signals** (emotion, willingness-to-share, difficulty) on **PARA only**.
- **C2 — Why aggregation works (the supporting science).** Individual reactions sit near the noise ceiling (Exp 0) and are barely predictable, but the **group aggregate is far more predictable** because idiosyncratic variance averages out. We report (i) the **aggregate-vs-individual predictability gap** (same model: individual error ≫ group error) and (ii) the **N-personas → fidelity curve** (group-prediction error falls and saturates as the panel grows). Individual cold-start prediction appears here only as the lower bound — *not* the headline. Collaborative filtering is shown only in the warm-start (seen-user) regime as a reference.
- **C3 — Generalization to generated images (cross-cultural).** All judging is **zero-shot with frozen serverless VLM calls**; the only thing tuned on real images is a post-hoc score→reaction calibration (no weights change). We test whether the group-prediction recipe predicts cross-cultural preference on public **Rapidata** votes sliced by **country/language**. Confounds handled head-on: (i) restrict to nationalities **present in LAPIS** so a real-image reference for the same group exists; (ii) compare against a **no-persona prompt** (isolates the persona signal) and a **global-preference** baseline (isolates universal quality). Decisive metric on **slice-disagreement pairs** only.
- **C4 — Useful, non-circular editing feedback.** Aggregated *group* critique is distilled into an edit instruction, applied by a **frozen serverless editor** (FLUX.1-Kontext provider route if available), and reranked by a **held-out evaluator** (EditReward if serverless-accessible; otherwise EditReward-Bench labels / serverless scalar evaluators). Two nested claims: **C4-core** — audience-*aggregate* critique beats single-judge and no-feedback (independent of slicing); **C4-targeted** — audience edits win *more within a targeted group*, **contingent on C3 showing non-trivial between-group variance**.

*Individual prediction is expected to be weak (Exp 0). That is the point: the contribution is that **group-level** prediction works where individual prediction cannot.*

### 2.1 Claim → data → experiment

| Claim | Data | Decisive test | Failure mode handled |
|---|---|---|---|
| **C1 group prediction** | PARA/LAPIS (real group ~25/img), pooled per slice | beat no-persona aggregate + population prior; **capture between-group gaps** | per-image-per-slice excluded (pool) |
| **C2 aggregation gap** | PARA/LAPIS individual vs. aggregate | individual error ≫ group error; **N-curve saturates** | individual bounded by Exp 0 ceiling |
| C2 structured | PARA only | per-signal group calibration | scoped to photos |
| **C3 generated (cross-cultural)** | Rapidata (country/language per vote) | slice-disagreement win-rate vs. no-persona + global | LAPIS-nationality subset + controls |
| **C4 editing** | EditReward-Data/Bench (+ frozen editor) | aggregate-critique win; targeted win | targeting contingent on C3 variance |

---

## 3. Data (only what the claims require)

> ✅ = on Hugging Face · ⚠️ = via request/GitHub.

**Persona anchors (C1/C2).** The only public sources with *per-rater* persona + reactions at scale — and, crucially, ~25 raters/image gives a **real group** per image to validate group prediction against.

| Dataset | Scale | Persona | Reactions | Access |
|---|---|---|---|---|
| **PARA** | 31,220 photos, 438 subjects, ~25/img | age, gender, education, **Big-5**, art/photo experience | score + emotion, content preference, willingness-to-share, difficulty | ⚠️ project page / request |
| **LAPIS** | 11,723 WikiArt paintings, 552 subjects, ~24/img | age, gender, **nationality**, education, art interest | 0–100 appeal | ⚠️ [GitHub](https://github.com/Anne-SofieMaerten/LAPIS) request |

*LAPIS nationality coverage gates the C3 cross-cultural slice — audit it in week 1.*

**Critique language (rationale + edit instructions; population-level, not persona-validated).** **ArtEmis** (via ✅ `youssef101/artelingo`, art-emotion explanations) and ⚠️ **RPCD** (Reddit photo critiques; project page) for constructive "what's weak and why" language. Used only as **optional few-shot in-context exemplars (no training)**.

**Generated-image regime (C3).** ✅ **Rapidata** `Rapidata/*` — 700k+ pairwise votes over FLUX/SD3.5/MJ/DALL·E 3, **country + language per vote** (no Big-5/age/gender), three axes (preference/coherence/alignment). Pairwise + coarse, so C3 is the cross-cultural slice, evaluated in pairwise space, pooled across pairs, top-N slices. ✅ **RichHF-18K** *(optional)* — fine-grained plausibility/alignment/aesthetic axes; an extra generated-image sanity check / few-shot exemplar pool (not required).

**Editing (C4).** ✅ **EditReward-Data** (200K expert preference pairs) + **EditReward-Bench** (multi-way human ranking over ~12 editors incl. flux_kontext) + **EditReward** model (Qwen2.5-VL-7B) if serverless-accessible; otherwise use the benchmark labels / serverless scalar evaluators as held-out non-circular signals.

**Optional study (enrichment only — paper is fully provable on the above).** ~300 generated images, ~40–60 de-identified raters with PARA/LAPIS-style intake, ≥10 raters/image (a real group), pooled subgroup endpoints; one IRB. Adds the *psychographic* group axis (Big-5, expertise) on generated images that Rapidata lacks, and a group-targeted edit-preference test. Run only if IRB clears.

---

## 4. Serverless backbones (HF Inference Providers)

The judge needs *affective/aesthetic* reasoning, so the **primary** judge is the strongest conversational VLM available through **Hugging Face Inference Providers** at experiment time. This is an API/runtime choice: the model weights remain frozen and hosted by the provider, and the code logs the exact `(model, provider, revision/policy, prompt, decoding params)` for every call. A model being present on the HF Hub is **not sufficient**; it must be callable through Inference Providers for the serverless MPR.

| Role | Choice | HF |
|---|---|---|
| Judge — primary (serverless default) | **Best available HF Inference Providers VLM**, pinned in week 1 before test access. Seed candidates to audit: `zai-org/GLM-4.5V`, `CohereLabs/aya-vision-32b:cohere`, and any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM. | ✅ if callable via Inference Providers |
| Judge — robustness (serverless panel) | 2-4 additional provider-backed VLMs, spanning providers/families where possible; re-run **C1 + steerability** only. | ✅ availability-gated |
| Offline/open-weight shortlist (not MPR) | Gemma 4, Qwen3-VL, Kimi-VL, Phi-4-multimodal, NVIDIA Nemotron Nano VL remain scientific candidates only if they become serverless-callable; otherwise they are excluded from the serverless-only result. | ⚠️ HF Hub presence != serverless support |
| Editing reranker + baseline | **EditReward** if available through a serverless route; otherwise use EditReward-Bench labels for evaluation and serverless scalar/evaluator baselines, keeping proposer != reranker. | availability-gated |
| Frozen editor | **FLUX.1-Kontext-dev** via HF Inference Providers/provider route when available; SD-3.5-Medium fallback via provider route. | ✅ availability-gated |

The **unified default judge is a serverless HF Inference Providers VLM**, not a locally loaded `transformers` model. For robustness we **re-run C1 + steerability on a compact serverless cross-model panel** to show results are not provider/model-specific. If a desired open-weight VLM is only downloadable from the Hub but not callable through Inference Providers, it is documented as a rejected candidate rather than silently moved to local GPU or a dedicated endpoint.

---

## 5. Method

**5.1 AudienceJudge (frozen, serverless).** Input = image + **persona card** (NL text: age bucket, education, art/photo experience, Big-5 where available, nationality). Output = scalar score, structured reactions (PARA schema), short rationale (population-level). Image is the only visual input. The judge is called through HF Inference Providers; **no backbone weights are downloaded, trained, fine-tuned, or deployed by us**.

**5.2 Frozen prediction recipe.** The method's only levers are prompting and post-processing: (1) **zero-shot** persona-conditioned prompting; (2) optional **few-shot in-context exemplars** (generic or retrieved nearest (image, persona, reaction) from a train pool); (3) optional **self-consistency** (sample several outputs, aggregate); (4) **post-hoc output calibration** (isotonic/temperature mapping fit on a held-out *real-image* calibration split — this maps scores only and changes **no model weights**). The same recipe, with calibration fixed on real images, is what we test for generalization to generated images (C3).

**5.3 Synthetic audience (the core object).** To simulate a group, define the group's persona distribution, **sample N persona cards** to match it, run the frozen judge per persona, and **aggregate** into the predicted group reaction distribution (and its mean/dispersion). This panel-and-aggregate step *is* the method for C1; N is swept in C2. No inter-agent influence in the main method; an optional peer-revision probe is reported as non-validated.

**5.4 Audience-guided editing (non-circular).** Aggregate the group's complaints → distill one edit instruction → frozen serverless editor produces candidates → **rerank/evaluate with held-out signals only** (EditReward if serverless-accessible, otherwise EditReward-Bench labels / serverless scalar evaluators; the proposer never reranks) → optional human study.

---

## 6. Experiments

**Exp 0 — Ceiling (run first).** Variance decomposition on PARA/LAPIS: between-rater vs. within-rater noise vs. attribute-explained variance, plus inter-rater reliability. This **bounds** individual-level prediction and *motivates aggregation*: it quantifies how much idiosyncratic variance the group average is expected to cancel.

**Steerability gate.** Permute/ablate persona attributes; predictions must shift consistently with **held-out subgroup differences measured in the data** (not with priors/stereotypes). Near-zero steerability ⇒ persona is non-functional ⇒ pivot the paper to that.

**C1 (group prediction — headline).** Predicted vs. observed **group reaction distribution** per image/slice (Wasserstein-1, KL, ECE; reliability diagrams), **pooled across images**. Baselines: **no-persona aggregate** and **population-mean prior**. The decisive sub-result is **between-group separation**: predicted group differences track observed differences across slices.

**C2 (why aggregation works).** (i) **Aggregate-vs-individual gap**: same model, report individual error (per-rater Spearman/MAE, vs. the Exp-0 ceiling) *next to* group error to show the order-of-magnitude difference. (ii) **N-personas curve**: group-prediction error vs. panel size N (expect monotone decrease + saturation). Warm-start CF shown here only, as a reference for the individual lower bound.

**C3 (generalization to generated).** On Rapidata, predicted vs. observed **pairwise win-rate on slice-disagreement pairs**, reported as **ΔAUC over the global-preference baseline** (so universal quality cannot inflate it), restricted to LAPIS-represented nationalities, with a **no-persona** control. Report the real→generated gap vs. C1.

**C4 (editing).** EditReward-Bench ranking agreement + win-rate: **C4-core** (aggregate vs. single-judge vs. no-feedback) and **C4-targeted** (within-group, contingent on C3 variance). Secondary: edit-drift (identity/structure preservation).

**Statistics.** Bootstrap CIs over raters and images; a small set of **pre-registered primary statistical endpoints**; Holm correction across the subgroup sweep.

**Leakage.** Artist-disjoint WikiArt splits + a memorization probe (can the model reproduce ArtEmis captions/titles?); Rapidata/generated regime is the leakage-free check.

**Ablations.** Persona-card fields; zero-shot vs. few-shot (generic vs. retrieved exemplars); self-consistency; with/without post-hoc calibration; **persona sampling scheme** (how the panel is drawn); **serverless judge/provider family** (primary VLM vs. additional HF Inference Providers VLMs); reranker source.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Individual taste barely predictable | *Expected* — that is C2's point; the headline is group, not individual |
| Group prediction trivially = global average | C1 must beat the **no-persona aggregate** and **capture between-group gaps**, not just match the mean |
| C3 confounded (universal quality / weak cross-cultural signal) | LAPIS-nationality subset + no-persona & global-preference controls + disagreement-conditioned ΔAUC |
| Frozen VLM barely uses persona | Steerability gate + few-shot exemplars + calibration; if effect is null, headline the ceiling finding |
| Stage-2 circularity (C4) | Proposer ≠ reranker; held-out EditReward/EditReward-Bench/serverless evaluator signal only |
| Cross-cultural stereotyping | Frame as dataset-sampled distributions, not national essences; report over/under-statement of group gaps; no deployment framing |
| PARA/LAPIS access / license / redistribution | Request + license audit week 1; release-safe artifacts only |
| HF Inference Providers availability/provider churn | Week-1 serverless audit; pin exact model/provider/policy before test access; keep rejected candidate log |
| Serverless cost/rate limits | Cache every `(image, persona, model, prompt)` call; async batching with backoff; run MPR before broad ablations |
| Private dataset images sent to hosted providers | License/privacy audit; use public or permissioned URLs/bytes only; do not send identifying rater data |
| 6-week scope | Ship the MPR first (below) |

---

## 8. Scope & timeline (~6 weeks to NeurIPS Creative AI, Aug 3 2026)

**Minimum Publishable Result:** Exp 0 + **C1 (group prediction)** + **C2 (aggregation gap + N-curve)** on **one pinned serverless HF Inference Providers VLM**. C3, C4, the cross-model robustness check, and the optional study are stretch goals layered on top.

- **Wk 1** — request PARA/LAPIS + license/nationality audit; pull HF data (ArtELingo, Rapidata, EditReward-Data/Bench); audit HF Inference Providers VLM support; stand up **frozen serverless** judge inference; pin primary model/provider before test access; (optional) IRB.
- **Wk 2** — Exp 0 ceiling; persona-card builder; prompt design; steerability gate.
- **Wk 3** — post-hoc calibration; **C1 group prediction + C2 aggregation gap/N-curve + baselines (MPR locks here).**
- **Wk 4** — **C3 on Rapidata** (zero-shot; disagreement ΔAUC + no-persona/global controls); leakage.
- **Wk 5** — editing loop + **C4 on EditReward-Bench**; serverless cross-model/provider robustness panel on C1+steerability; few-shot/self-consistency ablations.
- **Wk 6** — remaining ablations, bias diagnostics, writing.

**Venue.** Primary: NeurIPS Creative AI 2026 (non-archival, 2–6 pp). Method-forward variant for WiCV @ ECCV 2026 (personalization architecture, fairness, generated-image eval).

---

## 9. Defensible headline

> We do **not** claim to "simulate society," to predict individuals, or to train anything — the audience is a panel of off-the-shelf frozen VLM calls through HF Inference Providers. We claim that **aggregating persona-conditioned judgments approximates the measured reaction of a *group*** (where individual prediction is near-impossible — and we show why), that this group prediction generalizes to AI-generated images along measurable axes, and that the simulated audience is a useful, interpretable, non-circular feedback signal for editing.

---

### References (verify against primary sources)
PARA (arXiv 2203.16754) · LAPIS (arXiv 2504.07670) · ArtEmis (2101.07396)/ArtELingo (HF) · RPCD (2206.08614) · RichHF-18K/RAHF (2312.10240) · Rapidata T2I preference (HF `Rapidata/*`) · EditReward/-Data/-Bench (2509.26346, HF `TIGER-Lab/*`) · ImageReward · PickScore · HPS v2 · PAMELA (2604.07427, concurrent) · AesBiasBench (2509.11620) · individual differences in computational aesthetics (2502.20518) · HF Inference Providers · serverless VLM candidates pinned in week 1 · FLUX.1-Kontext-dev (HF/provider route) · SD-3.5-Medium (HF/provider route).
