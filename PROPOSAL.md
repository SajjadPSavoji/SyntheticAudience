# Synthetic Audiences for Creative AI: Persona-Conditioned Multimodal Judges that Predict Human Reactions and Guide Image Editing

## Overview

When we ask "is this image good?", we are not asking for a single number. Beauty and appeal are *collective* — a distribution of reactions across many viewers who differ in culture, training, and taste. This proposal asks whether we can **simulate that collective reaction** using off-the-shelf AI, and whether the simulated audience is useful enough to guide creative work.

Our approach is deliberately simple. We take a frozen vision-language model (VLM) — one we call through **Hugging Face Inference Providers**, without training, fine-tuning, or hosting any weights of our own — and prompt it to role-play many different viewers, one "persona" at a time. Each persona is described in plain text (age, education, art experience, personality, nationality). We run the model once per persona to get that viewer's predicted reaction to an image, then **aggregate** the many reactions into a predicted *group* reaction distribution. This panel-of-personas-then-aggregate step is the entire method.

The central thesis has three parts, which we defend in turn:

1. **The group is predictable even when the individual is not.** Individual taste is nearly random once you account for shared quality — but the *average* of a group cancels out that idiosyncratic noise, so the group reaction is far easier to predict. We show both that group prediction works and *why* it works.
2. **It generalizes from real images to AI-generated ones**, including across cultures, without changing any model weights.
3. **The simulated audience is a non-circular feedback signal for image editing** — we can turn the group's complaints into an edit instruction and verify the result with a held-out evaluator that never saw the critique.

**What makes this novel.** This is *not* another personalized-aesthetics model. We do not train a taste predictor. We use frozen, serverless VLM calls to predict the reaction of a *group* rather than one user; we demonstrate that aggregation is what makes the problem tractable; and we use the resulting audience as interpretable creative feedback. It also differs from text-only "silicon sampling" (simulating survey respondents with LLMs) because our audience reacts to *visual* stimuli.

---

## 1. Motivation

Aesthetic judgment is plural. The same photograph delights one viewer and bores another, and both reactions are valid — they belong to different people. So the honest target for a model of "how good is this image" is not a scalar score but the *shape* of the reaction across a population: its average, its spread, and how it shifts from one group of viewers to another.

Two questions follow directly. First, can we **simulate a group's reaction** to a given image, whether it is a real photograph, a painting, or something a generative model just produced? Second, if we can, is that simulated audience a *useful* signal — good enough to steer an editing loop toward images a real audience would prefer?

A key insight shapes everything: the unit of interest is the **group, not the individual**. We will show that individual taste sits close to the noise ceiling and is barely predictable, yet the group aggregate is predictable, because the idiosyncratic part of each person's reaction averages away. Making this contrast explicit — individual prediction is hard *by design*, group prediction is where the signal lives — is one of the paper's contributions, not a limitation to hide.

This fits the NeurIPS Creative AI **"Agency"** theme directly: when a system can anticipate how a group will react to an image, creative agency is redistributed among the audience, the model, and the creator.

**Positioning relative to recent work.**
- *PAMELA* (2026) personalizes a scalar reward to a *single* user and optimizes prompts against it. We instead model a *population* distribution, keep the generator frozen, and feed back natural-language critique rather than a scalar.
- *AesBiasBench* (2025) provides the lens we use to check whether our predictions are well-calibrated across demographic subgroups.
- Work on *individual differences in computational aesthetics* (2025) motivates our explicit "ceiling" analysis of how predictable individual taste can even be.

---

## 2. Claims

We make four claims. C1 is the headline; C2 explains why C1 is possible; C3 extends it to generated images; C4 shows it is useful.

### C1 — We can predict a group's reaction (headline)

If we run the frozen judge as a **panel of N personas sampled to match a target group** and aggregate their reactions, the resulting distribution matches the group's real reaction better than two natural baselines:
- a **no-persona judge** (the same model and prompt, but with an empty viewer profile), and
- the **population-mean prior** (just predicting the overall average for everyone).

We validate this on real images where each image was rated by a genuine group of roughly 25–35 people (the PARA, LAPIS, and EVA datasets), using distributional metrics — Wasserstein-1 distance, KL divergence, and expected calibration error — **pooled per group or slice**, never per-image-within-slice (individual images are too noisy for that). The decisive evidence is not matching the global average but **capturing the differences between groups**: does our model predict that, say, art experts and novices diverge on a given painting in the direction the data actually shows?

*Scope of C1.* We predict the **score** on photographs (PARA, EVA) and on art (LAPIS). We also predict **difficulty of judgment** on PARA and EVA (and, on EVA, the four aesthetic-attribute votes). The richer structured reactions — emotion, willingness to share, content preference — exist only in PARA, so those results are scoped to photographs.

### C2 — Why aggregation works (the supporting science)

C1 works because of a statistical fact we make measurable. Individual reactions sit near the noise ceiling (established in Exp 0) and are barely predictable, but the group aggregate is far more predictable because idiosyncratic variance averages out. We report this two ways:
- **The aggregate-vs-individual gap.** Using the *same* model, we place the individual-level prediction error next to the group-level error and show the group error is dramatically smaller.
- **The N-personas fidelity curve.** As we grow the panel from one persona to many, group-prediction error falls and then saturates. The saturation point tells us how many personas a faithful synthetic audience needs.

Individual cold-start prediction appears here only as the lower bound — it is *not* the headline. Collaborative filtering is shown only in the warm-start setting (predicting for a viewer we have already seen), purely as a reference point for the individual case.

### C3 — It generalizes to AI-generated images, across cultures

All judging is **zero-shot** with frozen serverless VLM calls. The only thing we ever fit on real images is a post-hoc mapping from the model's raw score to a calibrated reaction — this changes no model weights and is fixed before we touch generated images.

We then test whether the same recipe predicts **cross-cultural preferences on AI-generated images**, using public **Rapidata** votes sliced by the voter's country and language. Because cross-cultural claims are easy to confound, we handle two confounds head-on:
- **Universal quality.** We compare against a global-preference baseline, so "this image is just better" cannot masquerade as a cultural effect.
- **Persona signal.** We compare against a no-persona prompt, so we can isolate what the persona actually contributes.

We restrict the analysis to nationalities that also appear in LAPIS, so a real-image reference for the same group exists, and we score only on **slice-disagreement pairs** (image pairs where different groups genuinely prefer different images) — the only place a cross-cultural claim can be tested.

### C4 — The simulated audience is useful and non-circular editing feedback

We aggregate the group's complaints into a single edit instruction, apply it with a **frozen serverless image editor** (FLUX.1-Kontext via a provider route when available), and then judge the result with a **held-out evaluator that never proposed the edit**. This separation — the proposer is never the reranker — is what makes the signal non-circular. There are two nested claims:
- **C4-core:** aggregated *group* critique produces better edits than a single-judge critique or no feedback at all. This holds regardless of slicing.
- **C4-targeted:** edits guided by a specific group win *more within that group*. This one is contingent on C3 first showing that groups actually disagree.

> Individual prediction is expected to be weak (Exp 0). That is the point: the contribution is that **group-level** prediction succeeds where individual prediction cannot.

### 2.1 Claim → data → experiment at a glance

| Claim | Data | Decisive test | Failure mode handled |
|---|---|---|---|
| **C1 — group prediction** | PARA / LAPIS / EVA (real groups of ~25–35 raters/image), pooled per slice | beat the no-persona aggregate *and* the population prior; **capture between-group gaps** | per-image-within-slice noise (we pool) |
| **C2 — aggregation gap** | PARA / LAPIS / EVA, individual vs. aggregate | individual error ≫ group error; **N-curve saturates** | individual bounded by the Exp 0 ceiling |
| **C2 — structured signals** | emotion/willingness on PARA; difficulty on PARA + EVA | per-signal group calibration | scoped honestly to photos |
| **C3 — generated, cross-cultural** | Rapidata (country/language per vote) | win-rate on slice-disagreement pairs vs. no-persona and global baselines | LAPIS-nationality subset + controls |
| **C4 — editing feedback** | EditReward-Data / -Bench (+ frozen editor) | aggregate-critique win; targeted win | targeting contingent on C3 variance |

---

## 3. Data (only what the claims require)

Legend: ✅ available directly on Hugging Face · ⚠️ available by request or from GitHub.

### Persona anchors for C1 and C2

These are the backbone of the paper: the only public sources that pair **per-rater persona information** with **per-rater reactions** at scale, *and* collect enough raters per image (~25–35) to form a genuine group we can validate against.

| Dataset | Scale | Persona fields | Reactions collected | Access |
|---|---|---|---|---|
| **PARA** | 31,220 photos · 438 subjects · ~25/img | age, gender, education, **Big-5 personality**, art/photo experience | score + emotion, content preference, willingness-to-share, difficulty | ⚠️ [dataset page](https://web.xidian.edu.cn/ldli/en/dataset.html) → [Google Drive](https://drive.google.com/file/d/1ZKNceBy5eLn2XgPd2fsEQEosKfUkMdGO/view) (password-protected) |
| **LAPIS** | 11,723 WikiArt paintings · 552 subjects · ~24/img | age, gender, **nationality**, education, art interest | 0–100 appeal score | ⚠️ [homepage](https://sites.google.com/view/lapisdataset/homepage) → [OSF](https://osf.io/zw39r/files/osfstorage) (password-protected zip; password via [terms form](https://docs.google.com/forms/d/e/1FAIpQLSeEaohgR50NvgDBJ2h5ynsw_NOixWDuNNydpIOujm-wY4qE6g/viewform)) |
| **EVA** | 4,070 photos (an AVA subset) · ~30–40/img | age (birth year), gender, region, photography experience — *no Big-5, education, or art-familiarity* | integer score 0–10, four aesthetic-attribute votes (visual / composition / quality / semantic), difficulty, primary-factor label | ✅/⚠️ [GitHub](https://github.com/kang-gnak/eva-dataset) — CSV + AVA-subset images, **CC0-1.0** |

Three notes on how these fit together. **PARA** is the richest (it carries personality and the full structured-reaction schema) and covers photographs. **LAPIS** covers art and, crucially, records each rater's **nationality** — the field that gates the cross-cultural C3 slice, so we audit its coverage in week 1. **EVA** is a new addition that gives us a **third independent real-group photo cohort**: it is openly licensed (CC0, so redistribution is unrestricted, unlike PARA and LAPIS), it has *more* raters per image than PARA, and its difficulty rating maps cleanly onto PARA's. Its limitations are that it records only a coarse **region** (not nationality, so it does not feed C3) and lacks personality/education, and its images are drawn from AVA — so we deduplicate EVA images against any AVA-derived exemplars to avoid leakage.

### Critique language (for rationales and edit instructions)

To give the judge good vocabulary for *why* something works or fails, we optionally show it a few example critiques as in-context exemplars — never as training data. **ArtEmis** (via the ✅ `youssef101/artelingo` dataset) supplies art-emotion explanations, and ⚠️ **RPCD** (Reddit photo critiques) supplies constructive "what's weak and why" language.

### Generated-image regime for C3

✅ **Rapidata** (`Rapidata/*`) provides over 700,000 pairwise votes over images from FLUX, SD3.5, Midjourney, and DALL·E 3, with the voter's **country and language recorded per vote** across three axes (preference, coherence, alignment). It has no personality/age/gender, and it is pairwise rather than scalar — so C3 lives in pairwise space, pooled across pairs, on the top slices by volume. ✅ **RichHF-18K** is an optional extra source of fine-grained generated-image signals for sanity checks; it is not required.

### Editing for C4

✅ **EditReward-Data** (about 200K expert preference pairs) and **EditReward-Bench** (multi-way human rankings over ~12 editors, including FLUX-Kontext) provide the editing supervision and benchmark. The **EditReward** reward model (Qwen2.5-VL-7B) is used as the held-out evaluator *if* a serverless route exists; otherwise we evaluate C4 against the benchmark's human labels and serverless scalar evaluators, always keeping the proposer separate from the reranker.

### Optional human study (enrichment only)

The paper is fully provable on the datasets above. As a stretch goal, and only if IRB clears, we would collect ~300 generated images rated by ~40–60 de-identified participants with a PARA/LAPIS-style intake (≥10 raters/image, a real group). This would add the one thing Rapidata lacks — a **psychographic** group axis (Big-5, expertise) on generated images — plus a group-targeted edit-preference test.

---

## 4. Serverless backbones (Hugging Face Inference Providers)

Judging aesthetics requires affective and aesthetic reasoning, so the primary judge is the strongest conversational VLM available through **Hugging Face Inference Providers** at experiment time. This is a *runtime* choice, not a modeling one: the model's weights stay frozen and hosted by the provider, and our code logs the exact `(model, provider, revision/policy, prompt, decoding params)` for every call. A model merely being present on the HF Hub is **not enough** — it must be callable through Inference Providers to be part of the serverless result.

| Role | Choice | Availability |
|---|---|---|
| **Judge — primary** | The best available HF Inference Providers VLM, pinned in week 1 before any test access. Seed candidates to audit: `zai-org/GLM-4.5V`, `CohereLabs/aya-vision-32b:cohere`, and any newly provider-backed Gemma/Qwen/Kimi/Phi/Nemotron VLM. | ✅ if callable via Inference Providers |
| **Judge — robustness panel** | 2–4 additional provider-backed VLMs spanning different providers/families; we re-run only C1 and the steerability check on these. | ✅ availability-gated |
| **Offline open-weight shortlist** | Gemma, Qwen3-VL, Kimi-VL, Phi-4-multimodal, Nemotron Nano VL — candidates only *if* they become serverless-callable; otherwise excluded from the serverless-only result. | ⚠️ Hub presence ≠ serverless support |
| **Editing reranker + baseline** | EditReward via a serverless route if possible; otherwise EditReward-Bench labels and serverless scalar evaluators, always keeping proposer ≠ reranker. | availability-gated |
| **Frozen editor** | FLUX.1-Kontext-dev via a provider route when available; SD-3.5-Medium as fallback. | ✅ availability-gated |

The unified default judge is therefore a serverless VLM call, not a locally loaded model. To show our results are not an artifact of one model or provider, we re-run C1 and the steerability check on a compact cross-model panel. If a desired open-weight model turns out to be downloadable but not serverless-callable, we document it as a rejected candidate rather than quietly moving it onto a local GPU.

---

## 5. Method

### 5.1 The AudienceJudge

The core component is a frozen, serverless judge. Its input is an **image plus a persona card** — a short natural-language description of one viewer (age bucket, education, art/photo experience, Big-5 where available, nationality). Its output is a scalar score, the structured reactions from the PARA schema, and a short free-text rationale. The image is the only visual input. Every call goes through HF Inference Providers; **we download, train, fine-tune, and deploy no backbone weights.**

### 5.2 The prediction recipe (prompting and post-processing only)

The method's only levers are how we prompt the model and how we post-process its output — never the weights:
1. **Zero-shot persona-conditioned prompting** is the default.
2. **Few-shot in-context exemplars** (optional): we may prepend a handful of `(image, persona, reaction)` examples, either a fixed diverse set or nearest neighbours retrieved from the training pool, so the model can copy the mapping in context.
3. **Self-consistency** (optional): sample several outputs and aggregate them.
4. **Post-hoc calibration**: fit an isotonic or temperature mapping from raw to calibrated score on a held-out *real-image* split. This adjusts scores only, changes no weights, and is fixed before C3 — its generalization to generated images is itself part of the C3 claim.

### 5.3 The synthetic audience (the core object)

To simulate a group, we (a) define the group's persona distribution, (b) **sample N persona cards** to match it, (c) run the frozen judge once per persona, and (d) **aggregate** the results into a predicted group reaction distribution with its mean and spread. This panel-and-aggregate step *is* the method for C1, and N is the quantity we sweep in C2. The main method involves no interaction between personas; an optional peer-revision probe is reported separately as non-validated.

### 5.4 Audience-guided editing (kept non-circular)

For C4 we chain the pieces together: aggregate the group's complaints, distill them into a single edit instruction, apply it with the frozen serverless editor to produce several candidates, and then **rerank using held-out signals only** — the proposer never reranks its own edits. The optional human study provides an additional real-audience check.

---

## 6. Experiments

**Exp 0 — Ceiling analysis (run first).** Before trusting any prediction, we decompose the variance in PARA, LAPIS, and EVA into between-rater, within-rater, and attribute-explained components, and we measure inter-rater reliability. This *bounds* how predictable individual taste can be and quantifies how much idiosyncratic variance the group average is expected to cancel — which is exactly what motivates aggregation.

**Steerability gate.** We permute and ablate the persona fields and check that the judge's predictions shift in the direction the *data* says they should — not in the direction of stereotypes or priors. If the model essentially ignores the persona (near-zero steerability), the persona is non-functional, and we pivot the paper to reporting that finding.

**C1 — group prediction (headline).** We compare the predicted group distribution against the observed one (Wasserstein-1, KL, ECE, with reliability diagrams), pooled across images. Baselines are the no-persona aggregate and the population-mean prior. The decisive sub-result is **between-group separation**: predicted differences across slices must track the observed differences.

**C2 — why aggregation works.** We report (i) the aggregate-vs-individual gap — individual error placed next to group error for the same model, against the Exp 0 ceiling — and (ii) the N-personas curve, which should decrease and saturate. Warm-start collaborative filtering appears here only, as a reference for the individual lower bound.

**C3 — generalization to generated images.** On Rapidata, we compare predicted and observed pairwise win-rates on slice-disagreement pairs, reported as ΔAUC over the global-preference baseline (so universal quality cannot inflate the number), restricted to LAPIS nationalities, with a no-persona control. We also report the real-to-generated performance gap relative to C1.

**C4 — editing.** On EditReward-Bench we measure ranking agreement and win-rates for C4-core (aggregate vs. single-judge vs. no-feedback) and C4-targeted (within-group, contingent on C3 showing between-group variance). Secondary: edit drift (identity/structure preservation).

**Statistics.** Every headline number carries a bootstrap 95% CI, resampled over both raters and images. We fix a small set of primary statistical endpoints before test access and apply Holm correction across the subgroup sweep.

**Leakage.** We use artist-disjoint WikiArt splits and a memorization probe (can the model reproduce ArtEmis captions or titles?). The generated-image regime is inherently leakage-free and serves as a clean check.

**Ablations.** Persona-card fields; zero-shot vs. few-shot (generic vs. retrieved exemplars); self-consistency; with vs. without calibration; the persona-sampling scheme; the serverless judge/provider family; and the reranker source.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Individual taste is barely predictable | *Expected* — that is C2's whole point; the headline is the group, not the individual. |
| Group prediction is trivially just the global average | C1 must beat the **no-persona aggregate** and capture **between-group gaps**, not merely match the mean. |
| C3 is confounded by universal quality or a weak cross-cultural signal | LAPIS-nationality subset + no-persona and global-preference controls + disagreement-conditioned ΔAUC. |
| The frozen VLM barely uses the persona | Steerability gate + few-shot exemplars + calibration; if the effect is null, we headline the ceiling finding instead. |
| Stage-2 circularity in C4 | Proposer ≠ reranker; only held-out EditReward / EditReward-Bench / serverless-evaluator signals are used. |
| Cross-cultural stereotyping | Framed as dataset-sampled distributions, not national essences; we report over- and under-statement of group gaps; no deployment framing. |
| PARA/LAPIS access, license, or redistribution | Request and license audit in week 1; release only license-safe artifacts. (EVA is CC0, so it is unrestricted.) |
| HF Inference Providers availability / provider churn | Week-1 serverless audit; pin the exact model/provider/policy before test access; keep a rejected-candidate log. |
| Serverless cost and rate limits | Cache every `(image, persona, model, prompt)` call; async batching with backoff; run the MPR before broad ablations. |
| Sending dataset images to hosted providers | License/privacy audit; use only public or permissioned URLs/bytes; never send identifying rater data. |
| Six-week scope | Ship the Minimum Publishable Result first (below). |

---

## 8. Scope and timeline (~6 weeks to NeurIPS Creative AI, Aug 3 2026)

**Minimum Publishable Result (MPR):** Exp 0 + C1 (group prediction) + C2 (aggregation gap and N-curve) on a single pinned serverless VLM. C3, C4, the cross-model robustness panel, and the optional study all layer on top of this.

- **Week 1** — Request PARA/LAPIS and audit their licenses and LAPIS's nationality coverage; clone EVA; pull the HF data (ArtELingo, Rapidata, EditReward); audit which VLMs are actually serverless-callable; stand up the frozen serverless judge; pin the primary model/provider before test access; (optional) submit IRB.
- **Week 2** — Exp 0 ceiling; persona-card builder; prompt design; the steerability gate.
- **Week 3** — Post-hoc calibration; **C1 group prediction + C2 aggregation gap and N-curve + baselines.** *The MPR locks here.*
- **Week 4** — **C3 on Rapidata** (zero-shot; disagreement ΔAUC with no-persona and global controls); leakage checks.
- **Week 5** — The editing loop and **C4 on EditReward-Bench**; the cross-model/provider robustness panel on C1 and steerability; few-shot and self-consistency ablations.
- **Week 6** — Remaining ablations, bias diagnostics, and writing.

**Venue.** Primary: NeurIPS Creative AI 2026 (non-archival, 2–6 pages). A method-forward variant targets WiCV @ ECCV 2026 (personalization architecture, fairness, generated-image evaluation).

---

## 9. The defensible headline

> We do **not** claim to simulate society, to predict individuals, or to train anything — the audience is a panel of off-the-shelf frozen VLM calls through HF Inference Providers. We claim that **aggregating persona-conditioned judgments approximates the measured reaction of a *group*** (in a regime where predicting individuals is near-impossible — and we show why), that this group prediction **generalizes to AI-generated images** along measurable axes, and that the simulated audience is a **useful, interpretable, non-circular feedback signal** for editing.

---

### References (verify against primary sources)

PARA (arXiv 2203.16754) · LAPIS (arXiv 2504.07670) · EVA (Kang et al. 2020, doi:10.1145/3423268.3423590; [GitHub](https://github.com/kang-gnak/eva-dataset), CC0) · ArtEmis (2101.07396) / ArtELingo (HF) · RPCD (2206.08614) · RichHF-18K / RAHF (2312.10240) · Rapidata T2I preference (HF `Rapidata/*`) · EditReward / -Data / -Bench (2509.26346, HF `TIGER-Lab/*`) · ImageReward · PickScore · HPS v2 · PAMELA (2604.07427, concurrent) · AesBiasBench (2509.11620) · individual differences in computational aesthetics (2502.20518) · HF Inference Providers · serverless VLM candidates pinned in week 1 · FLUX.1-Kontext-dev (HF/provider route) · SD-3.5-Medium (HF/provider route).
