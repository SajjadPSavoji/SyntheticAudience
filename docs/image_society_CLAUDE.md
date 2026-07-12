# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Image society": a set of CSV-defined personas each independently react to an image, in
character, through a shared vision-language model (VLM) backend. Each persona returns a
0-100 "like" score and a short comment; results are printed and logged to JSON.

## Environment

Use the `persona` conda environment (not `base`) — it has `torch`, `transformers`, and `PIL`
installed:

```bash
conda activate persona
```

There is no requirements.txt/pyproject.toml; dependencies live only in the `persona` env.

## Running the pipeline

```bash
python src/pipeline.py [image] [--personas-csv data/personas.csv] \
    [--model-name Qwen/Qwen2-VL-7B-Instruct] [--limit N] [--output path.json]
```

or via the wrapper script:

```bash
bash script/pipeline.sh
```

`image` defaults to `data/images/img_001.jpg`. `--limit N` restricts to the first N personas
from the CSV — useful for a quick smoke test without waiting on all of them. Results are
always written to `data/logs/<image_stem>_<timestamp>.json` unless `--output` is given.

There is also a smaller demo, `script/opinions.py` (invoked via `bash script/test.sh`), which
hardcodes three personas and runs them against a procedurally generated image if none is
given. Unlike `pipeline.py`, it manually inserts `src/` onto `sys.path` because it lives in
`script/` rather than `src/`.

No automated test suite exists; "testing" currently means running the pipeline/demo scripts
against a real or generated image and inspecting the printed scores/comments.

## Dataset fidelity experiments (PARA)

`script/para_pipeline.py` (wrapper: `bash script/para_pipeline.sh`) asks whether VLM
personas built from *real* raters reproduce the ratings those raters actually gave. It
targets the PARA dataset (`data/PARA/annotation/PARA-Images.csv` — ~808k (image,
annotator) aesthetic scores on a 1-5 scale in 0.5 steps; `PARA-UserInfo.csv` — each
annotator's demographics + Big-Five profile; images at
`data/PARA/imgs/<sessionId>/<imageName>`).

```bash
python script/para_pipeline.py [--n-images 5] [--images name1.jpg,name2.jpg] \
    [--raters-per-image N] [--sampling stratified|uniform] [--seed 0] \
    [--temperature 0.0] [--backend qwen|llava] [--output path.json]
python script/para_pipeline.py --analyze-only data/logs/para_<ts>.json  # no GPU needed
```

For each sampled image it re-creates the ~25 annotators who really rated it and collects
in-character 1-5 scores, then reports (a) per-rating agreement (MAE/RMSE/correlation
against persona-blind baselines: predict-global-mean and predict-image-mean) and (b)
per-image distribution match (EMD/KS against a bootstrap-resampling noise floor). Logs go
to `data/logs/para_<timestamp>.json` (config, prompts, every raw response, metrics);
`--analyze-only` recomputes metrics from a log without loading the model.

Convention: dataset-specific prompts (persona-description builder, system prompt, scoring
question) live in the dataset's script, *not* in `src/persona` — rater features and score
scales differ per dataset (PARA is 1-5 halves; the generic pipeline is 0-100), so LAPIS /
eva-dataset (see `notebook/*_EDA.ipynb`) should get sibling scripts with their own prompts.

`--persona-blind` (all three dataset pipelines) is a model-side control: instead of
role-playing each real rater, it judges every image with one generic, rater-agnostic system
prompt (`*_GENERIC_SYSTEM_PROMPT`) and a de-personalized question, so you can see what the
VLM thinks *unconditioned on who is judging* and contrast it with the persona-conditioned
run and the statistical predict-mean baselines. The same (image, rater) tasks are still run
and scored against each rater's true value, so pair it with `--temperature > 0` (e.g. 0.7;
valid Qwen2-VL range is ~0-2, 0 = greedy) — otherwise every rater of an image gets a
byte-identical answer and the per-image distribution collapses to a point. The mode is
recorded as `persona_blind: true` in the log, and `--resume` refuses to mix blind and
persona ratings in one log.

`script/eva_pipeline.py` (wrapper: `bash script/eva_pipeline.sh`) is the EVA sibling: same
CLI and log/metrics shape, targeting `data/eva-dataset/` (~137k filtered ballots, ~30 per
image; voters have birth year, region, gender, photographic level, and eyesight — no Big
Five). It defines its own prompts and axes (`score` 0-10 integer, plus optional
`visual`/`composition`/`quality`/`semantic`/`difficulty` 1-4 via `--dimensions`), and
imports the dataset-agnostic machinery (ScoreDimension grids, response parsing, metrics,
image sampling) from `para_pipeline.py`. Images must first be extracted:
`cd data/eva-dataset/images && cat EVA_together.zip.00* > EVA_together.zip && unzip
EVA_together.zip` (yields `EVA_together/<image_id>.jpg`; already done on this machine).
Logs go to `data/logs/eva_<timestamp>.json`.

`script/lapis_pipeline.py` (wrapper: `bash script/lapis_pipeline.sh`) is the LAPIS
sibling: same CLI and log/metrics shape, targeting `data/LAPIS github/` (~284k PIAA
ratings of ~11.7k artworks/paintings, ~24 raters per image from 568 participants; raters
have age, nationality, gender, education, colour-blindness, and a VAIAK art-interest
score — no Big Five). Single axis: `rating`, the 0-100 aesthetic-appreciation slider.
Images live flat at `data/LAPIS github/images/<image_filename>` (a handful of
Latin-1-mangled filenames are missing on disk and are dropped at load time). ~6% of
ratings come from fully anonymous participants; they are excluded by default
(`--include-anonymous` runs them with a generic persona). Logs go to
`data/logs/lapis_<timestamp>.json`.

## Architecture

- `src/persona/backend/` — VLM backend abstraction. `base.VLMBackend` is an ABC with one
  method, `generate(system_prompt, image, prompt, max_new_tokens, **kwargs) -> str`. Each
  concrete backend (`qwen.QwenVLBackend`, `llava.LlavaBackend`) loads a HF model once in
  `__init__` and is meant to be shared across many `Person` instances rather than
  reconstructed per-persona. To add a new model family, subclass `VLMBackend` and export it
  from `backend/__init__.py`.
  - Note the backends are *not* interchangeable black boxes: LLaVA-1.5's chat template has no
    system slot and silently drops a `system` message, so `LlavaBackend.generate` folds the
    persona prompt into the user turn instead. Qwen2-VL supports a real system role. Keep this
    in mind when adding backends — check whether the chat template actually honors `system`.
- `src/persona/person.py` — `Person` turns a free-text `description` into a system prompt
  (`_system_prompt`) instructing the backend to role-play as that person reacting to an image
  in their social feed, then calls `backend.generate(...)`. The default question
  (`DEFAULT_QUESTION`) asks for strictly a single JSON object `{"score": int, "comment": str}`.
- `src/pipeline.py` — orchestration: loads personas from a CSV (`person_id`, `description`
  columns), builds one `Person` per row against a single shared backend, and parses each raw
  model response with `parse_rating`. Small VLMs sometimes wrap the JSON in markdown fences or
  add stray text, so parsing first tries `json.loads` on the substring between the first `{`
  and last `}`, then falls back to regexes (`_SCORE_RE`, `_COMMENT_RE`) over the raw text.
  Scores are always clamped to 0-100; an unparseable response yields `score=None` rather than
  raising, so one bad response doesn't abort the whole batch.
- `data/personas.csv` — the persona roster (`person_id,description`); each row becomes one
  `Person` sharing the same backend/model.
- `data/logs/*.json` — one file per pipeline run: image path, model name, personas CSV path,
  a UTC timestamp, and a `results` list with each persona's score, comment, and full
  `raw_response` (kept for debugging parse failures).
