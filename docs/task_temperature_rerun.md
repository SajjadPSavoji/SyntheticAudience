# Task: re-run the persona (`full`) ratings at temperature 0.7

**Owner:** _(teammate)_ · **Est. effort:** mostly GPU time; near-zero code (Option A) ·
**Prereq:** the same environment/model you used for the original runs (`Qwen/Qwen2-VL-7B-Instruct`,
the `persona` conda env, `script/*_pipeline.py`).

---

## TL;DR

The persona-conditioned runs (`para_full`, `eva_full`, `lapis_full`) were generated with
**greedy decoding (`--temperature 0.0`)**. The persona-blind controls were run at **0.7**. That
one difference is now our biggest confound and is **suppressing the persona signal we are trying
to measure**. Please **re-run the three `full` conditions at `--temperature 0.7`**, keeping every
other setting identical, and hand back the exported logs. No code change is required for the core
task (the pipeline already switches to sampling when temperature > 0).

---

## Why this matters (the reasoning — please read, it determines how to run it)

We audited the delivered runs and found that at temperature 0 the model's predictions **collapse
to a single value across personas** for about half of all images:

| run | fraction of images where every persona got the *identical* score |
|---|---|
| para_full (temp 0.0)  | **48%** |
| eva_full (temp 0.0)   | **51%** |
| lapis_full (temp 0.0) | 4% |
| *(all `blind` runs, temp 0.7, for reference)* | 86–90% collapse comes from a *generic* prompt, not decoding |

Three concrete problems follow, and each is fixed by sampling at temperature > 0:

1. **Greedy decoding quantizes away small persona effects.** Ratings are discrete (PARA
   integers/half-steps, EVA 0–10 integers, LAPIS 0–100 integers). Greedy decoding emits the single
   most-likely score — an argmax. If an art-expert persona shifts the model's internal preference
   from 3.2 → 3.4 and a novice sits at 3.1, **both argmax to "3"** and the persona effect is
   rounded to zero — invisible. Sampling at 0.7 and letting the panel spread turns that
   sub-threshold preference into a **frequency shift** (the expert lands on "4" more often), which
   is exactly the signal our steerability / between-group analyses need. This is why our current
   headline numbers (e.g. LAPIS between-group separation +0.17, PARA +0.04) are **floors** — the
   real effect is partly censored.

2. **The N-personas aggregation curve (claim C2) is meaningless at temp 0.** Aggregation works by
   *canceling variance* across the panel of personas. At temp 0 the panel has almost no variance,
   so there is nothing to cancel — our N-curve is flat and "saturates" at N≈2, which is a decoding
   artifact, not a property of aggregation. With temperature > 0 the panel has real spread and the
   curve can show the true decrease-and-saturate shape (and tell us how many personas we actually
   need).

3. **The group-distribution metrics (claim C1: Wasserstein / KL / ECE) need a real distribution.**
   Human groups disagree a lot; a synthetic panel that collapses to a near-point-mass cannot match
   the *shape* of the group's reaction, only (at best) its mean. Sampling gives the panel realistic
   dispersion to compare against the human distribution.

**Why 0.7 specifically:** the `blind` control was already run at 0.7. Matching `full` to 0.7 makes
the two conditions differ **only** in whether the persona is present — removing the temperature
confound so `full` vs `blind` becomes a clean persona-on / persona-off test. (0.7 is also a
sensible mid-range value for this model.)

---

## Option A — the core task (do this; no code change)

Re-run each `full` condition **exactly as before, changing only `--temperature 0.0` → `0.7`**.
The pipeline auto-enables sampling (`do_sample=True`) whenever temperature > 0, so nothing else
needs editing.

**Keep every other argument identical to your original `full` runs** so the new logs cover the
*same* (image, rater) tasks and stay comparable to both the old `full` (temp 0) and the existing
`blind` (temp 0.7): same `--model-name`, `--seed 0`, `--sampling`, `--n-images` / `--images`,
`--dimensions`, `--chunk-size`, and the same 4-way `--shard i/4` split.

Reference commands (adjust the shard loop / GPU pinning to your setup; **do not** pass
`--persona-blind` — this is the persona run):

```bash
# PARA  (see coordination note below before running PARA)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/para_pipeline.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct --seed 0 --sampling stratified \
    --n-images 2000 --temperature 0.7 --shard $i/4 --chunk-size 5000 \
    --output data/logs/para_full_t07.shard${i}of4.json &
done; wait

# EVA  (full dataset, 4070 images)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/eva_pipeline.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct --seed 0 \
    --temperature 0.7 --shard $i/4 --chunk-size 5000 \
    --output data/logs/eva_full_t07.shard${i}of4.json &
done; wait

# LAPIS
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/lapis_pipeline.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct --seed 0 \
    --n-images 4000 --temperature 0.7 --shard $i/4 --chunk-size 5000 \
    --output data/logs/lapis_full_t07.shard${i}of4.json &
done; wait
```

> Use the **exact same selection args** (`--n-images` / `--images`, `--seed`, `--sampling`) that
> produced the original `full` runs. If you kept the original commands, just copy them and swap the
> temperature and the output name. `--resume` works if a shard crashes.

---

## Option B — recommended enhancement (self-consistency; needs a small code change)

Option A samples **once** per (image, persona). That already fixes the panel-level collapse
(problems 2 and 3) and helps steerability, because our analysis averages many personas within each
group. To also de-quantize **each individual persona's** score (sharpening problem 1 further),
draw **T samples per (image, persona) and average the parsed scores** before writing `pred_*`.

Sketch (in each `*_pipeline.py`, around the generate call — search for
`gen_kwargs.update(do_sample=True, temperature=args.temperature)`):

- add a CLI flag `--samples T` (default 1);
- when `T > 1` and sampling is on, either pass `num_return_sequences=T` to `generate()` or loop the
  generate call `T` times for the batch;
- parse each of the `T` raw responses and store the **mean** of the numeric scores per dimension as
  `pred_<dim>` (keep one raw response for debugging).

Cost scales linearly with `T`. **T = 3–5 is a good default.** Treat Option B as optional: if GPU
budget is tight, ship Option A first — it unblocks the analysis on its own.

---

## Outputs & naming (so the analysis picks them up automatically)

- Name the runs **`para_full_t07`, `eva_full_t07`, `lapis_full_t07`** (add `_t07s5` etc. if you do
  Option B with 5 samples).
- Export/stitch the shards the same way the original results were delivered (per-shard summary
  `…shardIof4.json` + `…part-NNNN.json` chunks), and drop each run's files into its own folder:
  ```
  data/results/para_full_t07/…
  data/results/eva_full_t07/…
  data/results/lapis_full_t07/…
  ```
  (`data/results/` is git-ignored, so the files stay local — that's expected.)
- Hand back just those three folders (or ping when they're in place).

---

## Acceptance criteria (how we'll know it worked)

1. **Coverage matches.** Each `*_full_t07` run covers the **same (image, rater) set** as the
   corresponding old `full` / `blind` run (same seed/selection). We verify with
   `scripts/analysis/coverage.py` + `audit.py`.
2. **The collapse is gone.** In `scripts/analysis/audit.py` output, the `degen`
   (degenerate-prediction) fraction for the new runs should drop from ~0.48–0.51 toward a small
   value — the panel now has real spread.
3. **Signal comes off the floor.** Re-running `exp0`/`c2_ncurve`/`steerability`/`c1_separation`
   on the new logs should show a **non-flat N-curve** and **steerability / between-group separation
   ≥ the temp-0 values** (we expect LAPIS and PARA to rise; EVA may stay weak — that's an honest
   result about EVA's thin personas, not a bug).

_(Wiring the new folder names into `scripts/analysis/common.py` (the `RUNS` list) and re-running the
Tier-1 analyses is on me once the logs land.)_

---

## Coordination note — PARA

PARA image coverage is being widened separately (owner: the requester). To avoid running PARA
twice, please **do EVA and LAPIS first** (their sets are final — EVA is 100%, LAPIS is fixed at the
current 4000). For **PARA**, either:
- (preferred) run the temp-0.7 `full` **and** a matching temp-0.7-vs-blind pair on the **new wider
  PARA image set** once it's ready, so coverage and decoding are both final in one pass; or
- if you want to unblock analysis sooner, run temp-0.7 `full` on the **current 2000-image PARA set**
  now, and we'll re-do it on the wider set later.

Confirm which PARA path you're taking so we keep the (image, rater) sets aligned.
