# Architecture

How the code is put together, and the non-obvious decisions behind it. For *running* things
see [`REPRODUCING.md`](../REPRODUCING.md); for *what was found* see [`research_plan.md`](../research_plan.md).

---

## The two layers

```
src/persona/     the frozen synthetic audience  — a VLM role-plays one viewer at a time
src/editor/      the AutoPolish loop            — critic -> editor -> held-out judge
```

Everything else is a driver (`script/`) or tooling (`scripts/`) on top of these two.

---

## `src/persona` — the frozen judge

### `backend/` — VLM abstraction

`base.VLMBackend` is an ABC with one method:

```python
generate(system_prompt, image, prompt, max_new_tokens, **kwargs) -> str
```

Each concrete backend (`qwen.QwenVLBackend`, `llava.LlavaBackend`) loads a Hugging Face model
once in `__init__` and is **meant to be shared across many `Person` instances** rather than
reconstructed per persona — that sharing is what makes a 10-persona panel cheap. To add a model
family, subclass `VLMBackend` and export it from `backend/__init__.py`.

> **The backends are not interchangeable black boxes.** LLaVA-1.5's chat template has no system
> slot and *silently drops* a `system` message, so `LlavaBackend.generate` folds the persona
> prompt into the user turn instead. Qwen2-VL supports a real system role. When adding a
> backend, check whether its chat template actually honors `system` — a backend that drops it
> will look like it works while quietly running persona-blind.

### One image or several

`generate` and `generate_batch` accept `ImagesInput` — either a single image or a **sequence**
shown side by side in one turn. `load_images` normalizes both to a list, so single-image callers
are unaffected. Qwen2-VL interleaves them (one `{"type": "image"}` placeholder per image, bound
to the flattened batch in order); `LlavaBackend` has one `<image>` slot and now **raises** on a
multi-image turn rather than silently dropping all but the first.

This is what makes the pairwise Rapidata comparison possible — `rapidata_pipeline.py` requires
`--backend qwen` for exactly this reason.

### `person.py` — one viewer

`Person` turns a free-text `description` into a system prompt (`_system_prompt`) instructing the
backend to react to an image *as that person*, then calls `backend.generate(...)`. The default
question (`DEFAULT_QUESTION`) asks for strictly one JSON object
`{"score": int, "comment": str}`.

### Response parsing

Small VLMs wrap JSON in markdown fences or add stray prose. Parsing therefore tries, in order:
`json.loads` on the substring between the first `{` and the last `}`, then per-key regexes over
the raw text. Scores are clamped to their scale; an unparseable response yields `score=None`
rather than raising, so one bad response never aborts a batch. Observed parse rate is
essentially 100%.

### Prompt-location convention

**Dataset-specific prompts live in the dataset's driver script, not in `src/persona`.** Rater
features and score scales differ per dataset (PARA is 1–5 in half steps, EVA is 0–10 integer,
LAPIS is a 0–100 slider, the generic demo is 0–100), so each of
`script/{para,eva,lapis}_pipeline.py` defines its own persona-description builder, system
prompt, and scoring question. `src/persona` stays scale-agnostic.

---

## `src/editor` — the AutoPolish loop

```
critic.py       StaticCritic / BlindVLMCritic / SocietyCritic  + distill_instruction()
flux_editor.py  FluxKontextEditor / InstructPix2PixEditor      (build_editor)
objective.py    AestheticObjective   — LAION aesthetic predictor (CLIP + MLP head)
drift.py        DriftMetric          — DINOv2 cosine identity check
loop.py         run_refinement()     — anchored re-edit + accept-if-better
```

Two design rules are load-bearing for the paper's claims:

- **`critic != objective`.** The party that proposes an edit (`critic.py`, a Qwen VLM) is never
  the party that grades it (`objective.py`, a CLIP-based predictor + `drift.py`). Different
  model families, and the grader never sees the instruction — so "improvement" cannot be
  self-fulfilling, and no condition can win by writing an easy-to-follow instruction.
- **Anchored re-edit.** Each step applies the *accumulated* instruction to the **original**
  image, never to the previous output, so editing artifacts cannot compound over 10 steps.
  Combined with accept-if-better, the best-so-far trajectory is monotone by construction.

### Deferred imports

Importing `editor` pulls in **no** heavy dependencies (torch / diffusers / transformers).
Submodules are loaded lazily on first attribute access via PEP 562 (`__getattr__` in
`editor/__init__.py`). This lets the CPU-only analysis environment `import editor` without a GPU
stack installed. The dataset drivers use the same trick, deferring
`from persona import QwenVLBackend` until a GPU is actually needed.

If you add a public symbol to `editor`, register it in the `_LAZY` dict — it is the package's
export table.

---

## Drivers — `script/`

`script/{para,eva,lapis}_pipeline.py` ask the same question of three datasets: *do VLM personas
built from real raters reproduce the ratings those raters actually gave?* Each re-creates the
real annotators of a sampled image, collects in-character scores, and reports per-rating
agreement plus per-image distribution match against persona-blind and predict-the-mean
baselines.

Shared CLI across all three:

```
--n-images N          --raters-per-image N    --sampling stratified|uniform
--seed 0              --temperature 0.0       --backend qwen|llava
--persona-blind       --resume                --analyze-only <log.json>
--output path.json    --dimensions ...        (EVA only: extra 1-4 axes)
```

`--analyze-only` recomputes every metric from an existing log **without loading a model** — the
fast path for iterating on analysis.

`--persona-blind` is the model-side control: instead of role-playing each real rater, one
generic rater-agnostic system prompt judges every image. The same (image, rater) tasks still run
and are scored against each rater's true value.

> **Pair `--persona-blind` with `--temperature > 0`.** Under greedy decoding every rater of an
> image gets a byte-identical answer and the per-image distribution collapses to a point mass.
> This is the same decoding-collapse effect that flattens the panel-size curve on real images
> (paper §6, limitation 1). `--resume` refuses to mix blind and persona ratings in one log.

### Surviving a shared GPU

`generate_with_retry` (in `para_pipeline.py`, imported by the others) wraps every batch. On a
shared node, a neighbour ballooning its memory makes kernels fail from *inside* cuDNN/cuBLAS —
surfacing as `CUBLAS_STATUS_EXECUTION_FAILED` or a cuDNN graph assertion rather than a clean
torch OOM. Those failures are transient, so each attempt first **releases our own cached blocks**
(retrying while the caching allocator still holds the memory that starved the call is pointless),
waits with doubling backoff, then retries as a batch and finally one item at a time.

A rating that never got a real response is stored as a `<generation error: ...>` placeholder.
`is_generation_error` makes those **not count as completed work**: `--resume` re-runs them, and
merging logs prefers a real response over a placeholder. Without this, a transient fault would
silently freeze into a permanent hole in the dataset.

### The pairwise sibling

`script/rapidata_pipeline.py` is the odd one out: one persona choosing between **two** images
rather than scoring one. It replays real preference voters on AI-generated pairs, reconstructing
each from the only attributes Rapidata ships — country and language.

Three constraints in that data shape the design: the human annotators were **prompt-blind** (so
`--criterion preference` withholds the generation prompt too); there are **no rater IDs** (so a
persona is a *stratum*, country × language, not an individual); and ~51% of votes are anonymous
(excluded unless `--include-anonymous`). Sharding is **by pair, never by vote**, because every
metric needs a pair's whole ballot in one place.

### The editing drivers

`script/c4_refine.py` is the AutoPolish driver: it runs the four critic conditions through
`run_refinement` and writes per-step logs plus committed images under
`<output-root>/{logs,edits,analysis}/`.

`script/claim4_pipeline.py` is its predecessor, kept because the paper reports it as the
self-refine comparison: one *fixed* persona per image, edits chained on the previous output,
Qwen-Image-Edit instead of FLUX, and the critic also acting as the scorer. The three differences
that make `c4_refine.py` non-circular — a panel, an anchored re-edit, and a held-out judge from a
different model family — are exactly what this earlier version lacks.

`script/{pixel,webscrap}_pipeline.py` run the judge over unlabeled corpora (no ground truth), so
they yield panel distributions with nothing to score against.

Two small utilities: `script/opinions.py` (three hardcoded personas against a generated test
image — the zero-data smoke test) and `script/export_results.py` (stitch a sharded run log back
into one self-contained JSON).

---

## Log format

A run writes a **summary shard** `<run>.shardNof4.json` (config, dimensions, the persona card
text per `userId`, and a precomputed `metrics` block) plus **chunk files**
`<run>.shardNof4.part-KKKK.json` holding the per-`(image, user)` records.

`scripts/analysis/common.py::load_run` reassembles these into one tidy DataFrame and adds a
`<dim>_gt_norm` / `<dim>_pred_norm` column per dimension on a common `[0,1]` scale, so every
downstream analysis is scale-agnostic. `SCALES` and `PRIMARY_DIM` in that module are the single
source of truth for native rating ranges.

---

## Figures

`scripts/analysis/theme.py` installs one colorblind-validated palette used by every plotting
script, so the paper reads as a single visual system. Main-text composite figures are built by
`scripts/analysis/paper_figs.py`.

> Figures are drawn at their **final printed width (5.5in) with 6–7pt type** and included at
> `width=\linewidth`. Drawing wide and letting LaTeX scale down renders axis labels at ~3pt in
> print. `paper_figs.print_size()` sets the rcParams for this.

---

## Testing

There is no automated test suite. "Testing" means running a driver against real or generated
input and inspecting the output:

```bash
python script/opinions.py            # no data, no dataset — generates a test image
bash script/eva_pipeline.sh          # 5 images, seed 0
```

The analysis suite is self-checking in a weaker sense: every script re-derives its numbers from
the cached logs, so re-running `scripts/analysis/*.py` against a fetched `data/results/` should
reproduce the committed `results/*.json` values.
