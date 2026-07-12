# PARA pipeline — usage, sharding, resuming & growing a run

`script/para_pipeline.py` replays the **real** PARA annotators as VLM personas: for each
sampled image it re-creates the exact people who rated it, asks each one (in character) for a
1–5 aesthetic score, and measures how well the simulated scores reproduce the real ones
(per-rating MAE/RMSE/correlation vs. baselines, and per-image distribution match via EMD/KS
vs. a bootstrap noise floor).

This document covers how to **run it, split it across GPUs, resume after a crash, grow an
existing run with more images, and merge everything into final metrics**. For the research
question and dataset details see `CLAUDE.md`; this is the operational guide.

---

## 1. Prerequisites

- Run everything **from the repo root**.
- Use the `persona` conda environment (it has `torch`, `transformers`, `PIL`):

  ```bash
  conda activate persona
  ```

  or call the interpreter directly, e.g. `/shared/miniconda3/envs/persona/bin/python`.
- PARA data must be present under `data/PARA/` (`annotation/PARA-Images.csv`,
  `annotation/PARA-UserInfo.csv`, and images under `imgs/<sessionId>/<imageName>`).
- A GPU is needed to generate ratings. `--analyze-only` (metrics only) needs **no** GPU.

---

## 2. Quick start

```bash
# Smoke test: 5 images, all their real raters, deterministic
python script/para_pipeline.py --n-images 5 --seed 0

# A specific image, capped at 3 raters
python script/para_pipeline.py --images iaa_pub10_.jpg --raters-per-image 3

# Recompute metrics from an existing log — no model / no GPU
python script/para_pipeline.py --analyze-only data/logs/para_<timestamp>.json
```

Pin a run to one GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py --n-images 1000 --output data/logs/para_full.json
```

---

## 3. Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-images N` | `5` | Number of images to sample. |
| `--images a.jpg,b.jpg` | — | Run these exact images instead of sampling. |
| `--sampling stratified\|uniform` | `stratified` | `stratified` spreads picks over the human mean-score range and is **nested** (see §6); `uniform` is plain random and is **not** nested. |
| `--raters-per-image N` | all (~25) | Cap raters per image. Leave unset for distribution metrics. |
| `--seed N` | `0` | Sampling seed. Must be identical across shards / resumes / growth. |
| `--dimensions ...` | active axes | Which PARA score axes to elicit/evaluate. |
| `--backend qwen\|llava` | `qwen` | VLM backend family. |
| `--model-name` | `Qwen/Qwen2-VL-7B-Instruct` | HF model id. |
| `--batch-size N` | `8` | Ratings per fused `generate()` call. Higher = faster, more VRAM. |
| `--temperature` | `0.0` | `0` = greedy/deterministic. |
| `--shard i/N` | — | Run only image-shard `i` of `N` (see §4). |
| `--output PATH` | `data/logs/para_<ts>.json` | Summary log path. |
| `--resume LOG` | — | Continue a previous run from its summary log (see §5). |
| `--analyze-only LOG[,LOG...]` | — | Recompute/merge metrics only, no model (see §7). |
| `--chunk-size N` | `5000` | Max ratings per `.part-NNNN.json` chunk file. |
| `--checkpoint-interval S` | `60` | Seconds between in-progress checkpoints. |

---

## 4. Output file layout

A run does **not** write one big file. It writes:

- a small **summary** `<stem>.json` — config, personas, metrics, and a manifest of the chunk
  files;
- one or more **result chunks** `<stem>.part-NNNN.json` — the actual ratings and raw model
  responses, at most `--chunk-size` each.

The run is also **checkpointed every `--checkpoint-interval` seconds**, so a crash or kill
does not lose completed ratings.

### The `.shardIofN` suffix

> **If you pass `--shard i/N`, the script automatically inserts a `.shardIofN` suffix into
> your output name.** This is intentional, not a bug.

So:

```bash
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py --n-images 1000 --shard 0/2 --output data/logs/para_full.json
```

writes `data/logs/para_full.shard0of2.json` (+ `para_full.shard0of2.part-0000.json`, …).
You will **not** get a bare `para_full.json` while sharding — that name is produced only by
the merge step in §7. The suffix keeps parallel shard processes from clobbering each other's
files.

---

## 5. Multi-GPU: sharding

The workload is embarrassingly parallel **across images**. To use N GPUs, launch N processes,
each pinned to one GPU and given a disjoint image slice via `--shard i/N`. Sharding is at the
**image** level, so every rater of an image stays in the same shard and per-image
distribution metrics stay valid.

**Every process must share the same `--n-images`, `--seed`, and `--sampling`** — sharding
slices a single deterministic ordering, so identical selection args are what make the shards
disjoint halves of the *same* image set.

Example — 2 GPUs over 1000 images:

```bash
# GPU 7 — first half
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py \
    --n-images 1000 --shard 0/2 --output data/logs/para_full.json &

# GPU 6 — second half
CUDA_VISIBLE_DEVICES=6 python script/para_pipeline.py \
    --n-images 1000 --shard 1/2 --output data/logs/para_full.json &

wait
```

This produces `para_full.shard0of2.json` and `para_full.shard1of2.json` (each with their own
`.part-*` chunks). Merge them into final metrics with §7.

> **Common mistake:** running only `--shard 0/2` covers just **half** the images (the
> even-indexed slice `selected[0::2]`). You must also run `--shard 1/2` to cover the rest.

Same idea for 4 GPUs:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/para_pipeline.py \
    --n-images 31220 --shard $i/4 --output data/logs/para_full.json &
done
wait
```

---

## 6. Resuming a crashed or partial run

Point `--resume` at the run's **summary log** (the `<stem>.json`, not a `.part-*` file). The
script loads the ratings already present, skips their `(sessionId, imageName, userId)` tasks,
and appends new ones to the same files.

```bash
# resume a plain (unsharded) run
python script/para_pipeline.py --n-images 1000 --resume data/logs/para_<ts>.json

# resume one shard of a multi-GPU run
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py \
    --n-images 1000 --shard 0/2 --resume data/logs/para_full.shard0of2.json
```

**Rules:**

- `--resume` must be paired with the **same selection args** as the original run
  (`--n-images`, `--seed`, `--sampling`, `--raters-per-image`, `--dimensions`, and `--shard`
  if it was sharded). That's what makes the task list reproducible.
- Resume writes back into the **resumed file** — do not also pass a different `--output`.
- For a sharded run, resume **each shard's own summary file** separately.

---

## 7. Growing an existing run (more images)

Because the default **`stratified` sampling is nested** — `selection(N)` is always a subset of
`selection(N')` for `N' > N` — you can enlarge a finished run *without recomputing anything*.
`selection(2000)` is exactly `selection(1000)` plus 1000 brand-new images, in that order, and
the same holds within each shard slice. So you just **resume with the larger `--n-images`**;
the per-task dedup skips everything already done and runs only the new images.

Growing the 1000-image, 2-shard run above to 2000 images:

```bash
# shard 0 — resume, bump to 2000
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py \
    --n-images 2000 --shard 0/2 --resume data/logs/para_full.shard0of2.json

# shard 1 — resume, bump to 2000
CUDA_VISIBLE_DEVICES=6 python script/para_pipeline.py \
    --n-images 2000 --shard 1/2 --resume data/logs/para_full.shard1of2.json
```

Each prints `… ratings already done, N remaining` and only processes the ~1000 new images.

> **This only works with `--sampling stratified` (the default).** With `--sampling uniform`
> the set is redrawn from scratch and is **not** nested, so your old images would not be
> preserved — don't grow a uniform run this way. Also keep `--seed` unchanged.

---

## 8. Merging shards & computing final metrics

`--analyze-only` recomputes metrics from existing logs **without loading the model** (no GPU).
Pass multiple logs as a **single comma-separated argument** (no spaces) to merge them:

```bash
python script/para_pipeline.py --analyze-only \
    data/logs/para_full.shard0of2.json,data/logs/para_full.shard1of2.json \
    --output data/logs/para_full.json
```

- **One** log → updated in place (unless `--output` is given).
- **Several** comma-separated logs → results are combined and merged metrics are written to
  `--output` (or a `data/logs/para_merged_<ts>.json` file if `--output` is omitted).
- This merge step is what finally produces the plain `para_full.json` you use for reporting.

> **Note the comma.** `--analyze-only` takes one argument split on commas — space-separating
> the paths will not merge them.

---

## 9. End-to-end example (2 GPUs, 1000 → 2000 images)

```bash
conda activate persona
cd /path/to/image-society

# 1. Run both shards over 1000 images
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py --n-images 1000 --shard 0/2 --output data/logs/para_full.json &
CUDA_VISIBLE_DEVICES=6 python script/para_pipeline.py --n-images 1000 --shard 1/2 --output data/logs/para_full.json &
wait

# 2. Merge -> final metrics for 1000 images
python script/para_pipeline.py --analyze-only \
    data/logs/para_full.shard0of2.json,data/logs/para_full.shard1of2.json \
    --output data/logs/para_full.json

# 3. Later: grow to 2000 images (reuses the first 1000)
CUDA_VISIBLE_DEVICES=7 python script/para_pipeline.py --n-images 2000 --shard 0/2 --resume data/logs/para_full.shard0of2.json &
CUDA_VISIBLE_DEVICES=6 python script/para_pipeline.py --n-images 2000 --shard 1/2 --resume data/logs/para_full.shard1of2.json &
wait

# 4. Re-merge -> final metrics for 2000 images
python script/para_pipeline.py --analyze-only \
    data/logs/para_full.shard0of2.json,data/logs/para_full.shard1of2.json \
    --output data/logs/para_full.json
```

---

## 10. Gotchas checklist

- `--shard i/N` always adds a `.shardIofN` suffix; the bare `--output` name only appears from
  the `--analyze-only` merge.
- Running one shard of `N` covers only `1/N` of the images — launch all `N`.
- Keep `--n-images` (when sharding), `--seed`, and `--sampling` identical across shards,
  resumes, and merges.
- Resume from the **summary** `<stem>.json`, not a `.part-*` chunk, and per **shard** file.
- Growing a run only reuses prior work under `--sampling stratified` (the default).
- `--analyze-only` logs are **comma-separated in one argument**, not space-separated.
- `--analyze-only` needs no GPU; generation does.
```
