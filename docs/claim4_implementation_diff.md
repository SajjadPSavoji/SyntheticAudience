# Claim 4 — two different loops, and why the current results come out flat

*Comparison of `script/claim4_pipeline.py` (the run that produced `logs_claim4/`, 100 PARA
images × 10 iterations, Qwen-Image-Edit) against the loop design in the research plan
(implemented as `script/c4_refine.py`). Written after analyzing the `logs_claim4` numbers.*

## TL;DR

The two implementations are **not the same experiment**. `claim4_pipeline.py` is a **single-persona,
self-graded** refinement loop; the plan's C4 is a **critic-quality ablation with an independent
objective**. The mechanisms the plan relies on (independent evaluator, society/blind/static
conditions, accept-if-better, drift guardrail, anchored re-edit) are **absent** from
`claim4_pipeline.py` — which is exactly why its score trajectory is flat and dominated by regression
to the mean. This is a design gap, not a bug; the code runs correctly, it just can't test the claim
as written.

## What the numbers from `logs_claim4` show

| metric (persona's own 1–5 score) | value |
|---|---|
| mean Δ score, start → end | 2.99 → 3.09, **Δ +0.10, 95% CI [−0.04, +0.24]** (includes 0) |
| improved / flat / declined | 23% / **60%** / 17% |
| per-step changes up / flat / down | 7% / **86%** / 7% |
| corr(start score, Δ) | **−0.70** |
| Δ by start score | 1→+1.75, 2→+0.85, 3→+0.03, 4→**−0.50** |

The gain is not significant, most images/steps don't move, and the pattern is textbook **regression
to the mean** (low starts rise, high starts fall, everything converges toward ~3).

## Mechanism-by-mechanism difference

| Mechanism | `claim4_pipeline.py` | Plan / `c4_refine.py` |
|---|---|---|
| **Evaluator** | the **same persona VLM** that proposes the edit also scores it (`aestheticScore`) — proposer == evaluator | an **independent held-out model** (LAION aesthetic on CLIP, different family from the Qwen critic): *critic ≠ objective* |
| **Conditions** | one condition (single persona) | **society vs blind-VLM vs static** (+ reward-only oracle) — the actual claim is society > blind > static |
| **Panel** | one random annotator per image | a **panel of N personas** aggregated (that's the "society") |
| **Accept edits?** | every edit applied **unconditionally**; score can (and does) drop | **accept-if-better**: keep a candidate only if the objective improves |
| **Identity guard** | none | **drift guardrail** (DINOv2 cosine ≥ cap) rejects edits that wander too far |
| **Loop topology** | edits **compound on the previous output** (sequential) | **anchored re-edit**: apply the accumulated instruction to the *original* image (no artifact stacking) |
| **Candidates/round** | 1 | **K candidates**, keep the best feasible |
| Editor | Qwen-Image-Edit-2511 | FLUX.1-Kontext (either is fine) |

(`grep` over `claim4_pipeline.py` for `drift|clip|aesthetic model|society|panel|blind|accept|
held-out|objective` returns nothing but one docstring line.)

## Why the code produced exactly these results

Each missing mechanism maps directly to a symptom in the table above:

1. **No independent objective → the trajectory is circular.** The only signal is the persona
   re-rating its own edits on a coarse 1–5 half-point grid at temperature 0. That score is sticky
   (86% of steps don't change) and, being self-generated, can't distinguish "the image got better"
   from "the model re-asserted its number." A rising self-score would not prove improvement; here it
   doesn't even rise.
2. **No accept-if-better → 17% of images end up worse.** Nothing rejects a bad edit, so the loop is a
   random walk over the editor's outputs rather than an optimizer.
3. **No guardrail + compounding edits → uncontrolled drift** over 10 rounds, with no check that the
   image still resembles the source.
4. **Single persona, no conditions → the claim is untestable.** "Society gives better feedback than a
   blind VLM" needs at least two conditions to compare; the run has one.
5. **corr(start, Δ) = −0.70** is the tell: the apparent gains are a noisy self-rating regressing
   toward its mean (~3), not editing quality. Aggregated over 100 images it nets out to ≈0.

## What this means / suggested path

`claim4_pipeline.py` is a clean, working **plumbing demo** (100 images × 10 Qwen-Image-Edit rounds
complete end-to-end) — that part is valuable and reusable. But to *prove* Claim 4 we need the plan's
design: an **independent objective**, the **society/blind/static** conditions, and
**accept-if-better + drift** so "better" is measured by something other than the proposer.

`script/c4_refine.py` already implements all of that (society/blind/static critics, LAION-aesthetic
objective, DINOv2 drift, accept-if-better, anchored re-edit, K candidates). Two easy ways to
converge:

- **Reuse your editor:** port the `QwenImageEditor` wrapper from `claim4_pipeline.py` into
  `c4_refine.py` as an `--editor qwenedit` option, so we keep Qwen-Image-Edit but gain the missing
  mechanisms.
- **De-circularize the existing run cheaply:** re-score the already-saved `iter_*.jpg` images with the
  independent aesthetic + drift metrics (no persona in the loop) to see whether the edits improved on
  a neutral yardstick — a quick check on the data we already have.

Happy to do either.
