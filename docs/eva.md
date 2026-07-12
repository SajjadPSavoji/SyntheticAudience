# EVA pipeline — usage, sharding, resuming & growing a run

`script/eva_pipeline.py` replays the **real** EVA voters as VLM personas: for each sampled
image it re-creates the people who actually rated it, asks each one (in character) for a
0–10 aesthetic score (and optionally the 1–4 attribute ratings), and measures how well the
simulated scores reproduce the real ones (per-rating MAE/RMSE/correlation vs. baselines, and
per-image distribution match via EMD/KS vs. a bootstrap noise floor).

It is the EVA sibling of `script/para_pipeline.py` and shares the same CLI, log layout, and
metrics shape — it imports the dataset-agnostic machinery (ScoreDimension grids, JSON
parsing, metrics, image sampling, chunk IO) from `para_pipeline.py` and defines only the
EVA-specific prompts, axes, and data loading. **If you already know the PARA pipeline, this
document will look familiar by design.**

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
- EVA data must be present under `data/eva-dataset/`:
  - tables `data/eva-dataset/data/{votes_filtered.csv, users.csv, region_index.csv}`
    (these are `=`-separated CSVs);
  - **images extracted** to `data/eva-dataset/images/EVA_together/<image_id>.jpg`. They ship
    as a split zip and must be unpacked once (already done on this machine):

    ```bash
    cd data/eva-dataset/images && cat EVA_together.zip.00* > EVA_together.zip && unzip EVA_together.zip
    ```
- A GPU is needed to generate ratings. `--analyze-only` (metrics only) needs **no** GPU.

---

## 2. Quick start

```bash
# Smoke test: 5 images, all their real voters, deterministic
python script/eva_pipeline.py --n-images 5 --seed 0

# A specific image (image_id = AVA name without .jpg), capped at 3 voters
python script/eva_pipeline.py --images 71 --raters-per-image 3

# Elicit the attribute axes too, not just the overall score
python script/eva_pipeline.py --n-images 5 --dimensions score,visual,composition

# Recompute metrics from an existing log — no model / no GPU
python script/eva_pipeline.py --analyze-only data/logs/eva_<timestamp>.json
```

Or via the wrapper (which defaults to `--n-images 5 --seed 0` and forwards everything else):

```bash
bash script/eva_pipeline.sh --batch-size 16
```

Pin a run to one GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py --n-images 1000 --output data/logs/eva_full.json
```

---

## 3. Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-images N` | `5` | Number of images to sample. |
| `--images a,b` | — | Run these exact `image_id`s (AVA names without `.jpg`) instead of sampling. |
| `--sampling stratified\|uniform` | `stratified` | `stratified` spreads picks over the human mean-score range and is **nested** (see §7); `uniform` is plain random and is **not** nested. |
| `--raters-per-image N` | all (~30) | Cap voters per image. Leave unset for distribution metrics. |
| `--dimensions ...` | `score` + all attributes | Which EVA axes to elicit/evaluate. `score` is 0–10 integer; `visual`/`composition`/`quality`/`semantic`/`difficulty` are 1–4. |
| `--seed N` | `0` | Sampling seed. Must be identical across shards / resumes / growth. |
| `--backend qwen\|llava` | `qwen` | VLM backend family. |
| `--model-name` | `Qwen/Qwen2-VL-7B-Instruct` | HF model id. |
| `--batch-size N` | `8` | Ratings per fused `generate()` call. Higher = faster, more VRAM; `1` = one-at-a-time. |
| `--temperature` | `0.0` | `0` = greedy/deterministic. |
| `--max-new-tokens` | `128` | Generation budget per rating. |
| `--shard i/N` | — | Run only image-shard `i` of `N` (see §5). |
| `--output PATH` | `data/logs/eva_<ts>.json` | Summary log path. |
| `--resume LOG` | — | Continue a previous run from its summary log (see §6). |
| `--analyze-only LOG[,LOG...]` | — | Recompute/merge metrics only, no model (see §8). |
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
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py --n-images 1000 --shard 0/2 --output data/logs/eva_full.json
```

writes `data/logs/eva_full.shard0of2.json` (+ `eva_full.shard0of2.part-0000.json`, …).
You will **not** get a bare `eva_full.json` while sharding — that name is produced only by
the merge step in §8. The suffix keeps parallel shard processes from clobbering each other's
files.

---

## 5. Multi-GPU: sharding

The workload is embarrassingly parallel **across images**. To use N GPUs, launch N processes,
each pinned to one GPU and given a disjoint image slice via `--shard i/N`. Sharding is at the
**image** level, so every voter of an image stays in the same shard and per-image
distribution metrics stay valid.

**Every process must share the same `--n-images`, `--seed`, and `--sampling`** — sharding
slices a single deterministic ordering, so identical selection args are what make the shards
disjoint slices of the *same* image set.

Example — 2 GPUs over 1000 images:

```bash
# GPU 7 — first slice
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py \
    --n-images 1000 --shard 0/2 --output data/logs/eva_full.json &

# GPU 6 — second slice
CUDA_VISIBLE_DEVICES=6 python script/eva_pipeline.py \
    --n-images 1000 --shard 1/2 --output data/logs/eva_full.json &

wait
```

This produces `eva_full.shard0of2.json` and `eva_full.shard1of2.json` (each with their own
`.part-*` chunks). Merge them into final metrics with §8.

> **Common mistake:** running only `--shard 0/2` covers just **half** the images (the
> round-robin slice `selected[0::2]`). You must also run `--shard 1/2` to cover the rest.

Same idea for 4 GPUs:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python script/eva_pipeline.py \
    --n-images 4000 --shard $i/4 --output data/logs/eva_full.json &
done
wait
```

---

## 6. Resuming a crashed or partial run

Point `--resume` at the run's **summary log** (the `<stem>.json`, not a `.part-*` file). The
script loads the ratings already present, skips their `(imageName, userId)` tasks, and
appends new ones to the same files.

```bash
# resume a plain (unsharded) run
python script/eva_pipeline.py --n-images 1000 --resume data/logs/eva_<ts>.json

# resume one shard of a multi-GPU run
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py \
    --n-images 1000 --shard 0/2 --resume data/logs/eva_full.shard0of2.json
```

**Rules:**

- `--resume` must be paired with the **same selection args** as the original run
  (`--n-images`, `--seed`, `--sampling`, `--raters-per-image`, `--dimensions`, and `--shard`
  if it was sharded). That's what makes the task list reproducible. Resuming with a different
  `--dimensions` is rejected outright.
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
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py \
    --n-images 2000 --shard 0/2 --resume data/logs/eva_full.shard0of2.json

# shard 1 — resume, bump to 2000
CUDA_VISIBLE_DEVICES=6 python script/eva_pipeline.py \
    --n-images 2000 --shard 1/2 --resume data/logs/eva_full.shard1of2.json
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
python script/eva_pipeline.py --analyze-only \
    data/logs/eva_full.shard0of2.json,data/logs/eva_full.shard1of2.json \
    --output data/logs/eva_full.json
```

- **One** log → updated in place (unless `--output` is given).
- **Several** comma-separated logs → results are combined (deduplicated by
  `(imageName, userId)`) and merged metrics are written to `--output` (or a
  `data/logs/eva_merged_<ts>.json` file if `--output` is omitted).
- This merge step is what finally produces the plain `eva_full.json` you use for reporting.

> **Note the comma.** `--analyze-only` takes one argument split on commas — space-separating
> the paths will not merge them.

---

## 9. End-to-end example (2 GPUs, 1000 → 2000 images)

```bash
conda activate persona
cd /path/to/image-society

# 1. Run both shards over 1000 images
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py --n-images 1000 --shard 0/2 --output data/logs/eva_full.json &
CUDA_VISIBLE_DEVICES=6 python script/eva_pipeline.py --n-images 1000 --shard 1/2 --output data/logs/eva_full.json &
wait

# 2. Merge -> final metrics for 1000 images
python script/eva_pipeline.py --analyze-only \
    data/logs/eva_full.shard0of2.json,data/logs/eva_full.shard1of2.json \
    --output data/logs/eva_full.json

# 3. Later: grow to 2000 images (reuses the first 1000)
CUDA_VISIBLE_DEVICES=7 python script/eva_pipeline.py --n-images 2000 --shard 0/2 --resume data/logs/eva_full.shard0of2.json &
CUDA_VISIBLE_DEVICES=6 python script/eva_pipeline.py --n-images 2000 --shard 1/2 --resume data/logs/eva_full.shard1of2.json &
wait

# 4. Re-merge -> final metrics for 2000 images
python script/eva_pipeline.py --analyze-only \
    data/logs/eva_full.shard0of2.json,data/logs/eva_full.shard1of2.json \
    --output data/logs/eva_full.json
```

---

## 10. Gotchas checklist

- Images must be **extracted** to `images/EVA_together/<image_id>.jpg` first (§1).
- `--images` values are `image_id`s (AVA names **without** `.jpg`), e.g. `71,106`.
- `--shard i/N` always adds a `.shardIofN` suffix; the bare `--output` name only appears from
  the `--analyze-only` merge.
- Running one shard of `N` covers only `1/N` of the images — launch all `N`.
- Keep `--n-images` (when sharding), `--seed`, `--sampling`, and `--dimensions` identical
  across shards, resumes, and merges.
- Resume from the **summary** `<stem>.json`, not a `.part-*` chunk, and per **shard** file.
- Growing a run only reuses prior work under `--sampling stratified` (the default).
- `--analyze-only` logs are **comma-separated in one argument**, not space-separated.
- `--analyze-only` needs no GPU; generation does.
