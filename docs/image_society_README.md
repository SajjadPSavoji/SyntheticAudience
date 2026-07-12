# image-society

Show an image to a set of CSV-defined personas and see how each one reacts. Every persona is
a free-text description turned into a system prompt for a shared vision-language model (VLM),
which replies in character with a 0-100 "like" score and a short comment.

## Setup

```bash
conda activate persona
```

## Usage

```bash
python src/pipeline.py [image] [--personas-csv data/personas.csv] \
    [--model-name Qwen/Qwen2-VL-7B-Instruct] [--limit N] [--output path.json]
```

or:

```bash
bash script/pipeline.sh
```

`image` defaults to `data/images/img_001.jpg`. `--limit N` runs only the first N personas from
the CSV, useful for a quick check. Each run prints every persona's score/comment and writes a
JSON log to `data/logs/<image_stem>_<timestamp>.json`.

A smaller demo with three hardcoded personas is available via:

```bash
bash script/test.sh
```

## Personas

`data/personas.csv` holds the roster as `person_id,description` rows. Add or edit rows to
change who reacts to the image.

## Layout

```
src/persona/       Person + VLM backend abstraction (Qwen2-VL, LLaVA)
src/pipeline.py    CLI: run every persona against an image, log results
script/            Shell wrappers and a small demo script
data/personas.csv  Persona roster
data/images/       Sample input images
data/logs/         JSON output per run
```
