"""LAPIS fidelity pipeline: replay real LAPIS participants as VLM personas.

The LAPIS dataset (data/LAPIS github/) couples ~11.7k painting/artwork images
with ~284k individual PIAA ratings (LAPIS_PIAA.csv): one row per
(participant, image), an aesthetic-appreciation score on a 0-100 slider,
and the rater's demographics (age, nationality, gender, education,
colour-blindness) plus their VAIAK art-interest score. Each image has ~24
raters drawn from 568 participants.

For each sampled image this script re-creates *the exact participants who
rated it* as VLM personas and asks each one, in character, for the same
0-100 score the study collected. Fidelity is then measured at two levels:

1. per-rating — does the simulated person give the same score as the real
   person on the same image? (MAE / RMSE / correlation, against baselines)
2. per-image  — do the ~24 simulated raters reproduce the *score
   distribution* of the 24 human raters? (EMD / KS vs a bootstrap noise floor)

Per the repo convention, LAPIS-specific prompts (persona-description builder,
system prompt, scoring question, score axis) live here, not in src/persona:
LAPIS raters have different features than PARA annotators (nationality and
VAIAK art interest; no Big Five) and a different scale (0-100 slider vs 1-5
halves), and the images are artworks rather than photographs. The
dataset-agnostic machinery (ScoreDimension grids, JSON parsing, metrics,
stratified sampling) is imported from the sibling script/para_pipeline.py.

~6% of ratings come from participants who skipped the demographic survey
entirely; they are excluded by default (no persona can be built) unless
--include-anonymous is given, which runs them with a generic description.

Ratings are generated in batches (--batch-size, one fused model.generate call
per batch) rather than one persona at a time, which is the main throughput lever
on a GPU.

A full-dataset run is one generate() call per ballot (~284k), so fan it out
across GPUs with --shard i/N: each of N processes takes a disjoint slice of the
images (all raters of an image stay in one shard, keeping per-image distribution
metrics valid) and writes its own .shardIofN log. E.g. over 4 GPUs:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python script/lapis_pipeline.py \
        --n-images 11700 --shard $i/4 --output data/logs/lapis_full.json &
    done

yields data/logs/lapis_full.shard0of4.json,...,lapis_full.shard3of4.json, which
--analyze-only reads together for combined metrics.

Each run writes a small summary log (config, personas, metrics, and a manifest
of its result-chunk files) plus the ratings themselves in fixed-size
<stem>.part-NNNN.json chunks. The log is checkpointed *while running* (every
--checkpoint-interval seconds, default 60), so a crash or kill loses at most one
interval; resume where it stopped with:

    python script/lapis_pipeline.py ... --resume data/logs/lapis_<ts>.json

--resume must be paired with the same selection args (--n-images/--images/
--shard/--seed/--dimensions/--include-anonymous) as the original run; the
(imageName, userId) ratings already in the log are skipped and only the rest are
generated.

Run from the repo root (persona conda env):

    python script/lapis_pipeline.py --n-images 5 --seed 0 --batch-size 8
    python script/lapis_pipeline.py --images boris-kustodiev_shells-1918.jpg --raters-per-image 3
    python script/lapis_pipeline.py --analyze-only data/logs/lapis_<ts>.json

--analyze-only recomputes metrics from such a log (or several) without touching
the model.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` (persona package) and ``script`` (para_pipeline) importable when
# this file is run from anywhere, not just via ``python script/lapis_pipeline.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "script"))

from para_pipeline import (  # noqa: E402  (needs sys.path tweak above)
    DEFAULT_CHUNK_SIZE,
    ScoreDimension,
    _atomic_write_json,
    _flush_chunks,
    _n_parts,
    _parse_shard,
    _part_path,
    _print_dimension_summary,
    _read_log_and_results,
    _shard_suffix,
    choose_images,
    compute_metrics,
    parse_para_rating as parse_rating,  # generic over ScoreDimensions
)

LAPIS_ANNOTATION_DIR = REPO_ROOT / "data" / "LAPIS github" / "annotation"
LAPIS_IMAGE_DIR = REPO_ROOT / "data" / "LAPIS github" / "images"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

# --------------------------------------------------------------------------
# LAPIS score axes (columns of LAPIS_PIAA.csv)
# --------------------------------------------------------------------------
# PIAA carries a single rating axis: aesthetic appreciation on a 0-100 slider
# (integers in the data). The registry keeps the same shape as PARA/EVA so a
# future axis (or a rescaled variant) slots in without touching the plumbing.

LAPIS_DIMENSIONS: dict[str, ScoreDimension] = {
    "rating": ScoreDimension(
        "rating", 0.0, 100.0, 1.0,
        "aesthetic appreciation — how aesthetically pleasing this person finds "
        "the artwork (0 = not at all, 50 = middling, 100 = extremely)",
    ),
}

# Which axes to elicit and evaluate by default; the first entry is the primary
# axis used for image stratification and the progress readout. Override at
# runtime with --dimensions.
ACTIVE_DIMENSIONS = ["rating"]


def resolve_dimensions(keys: list[str]) -> list[ScoreDimension]:
    """Map a list of LAPIS axis keys to ScoreDimension objects, validating each."""
    if not keys:
        raise ValueError("no score dimensions selected")
    unknown = [k for k in keys if k not in LAPIS_DIMENSIONS]
    if unknown:
        raise ValueError(
            f"unknown score dimension(s): {unknown}; "
            f"choose from {list(LAPIS_DIMENSIONS)}"
        )
    return [LAPIS_DIMENSIONS[k] for k in keys]


# --------------------------------------------------------------------------
# LAPIS-specific prompts
# --------------------------------------------------------------------------
# The raters saw only the artwork itself (no artist, title, or style label),
# so the persona is shown the bare image too.

LAPIS_SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing as one specific human participant in an online study "
    "of aesthetic appreciation of artworks. The participant is {description}\n\n"
    "You are shown one artwork (a painting or drawing) from the study. Judge it "
    "the way this exact person would: let their age, nationality, gender, "
    "education, and level of interest in art shape both what they notice and "
    "how critical or generous they are. People differ widely in taste — some "
    "love abstract art, others only figurative work; not every artwork deserves "
    "a good score, so be honest when a piece would bore or leave this person "
    "cold. Do not mention that you are an AI or that you are role-playing."
)

# Persona-blind baseline: judge the artwork with no rater conditioning at all,
# so the model's generic opinion can be contrasted with the persona-conditioned
# runs (and the statistical predict-mean baselines). Selected with --persona-blind.
LAPIS_GENERIC_SYSTEM_PROMPT = (
    "You are an impartial, general-purpose judge of artistic aesthetics.\n\n"
    "You are shown one artwork (a painting or drawing) from a study of aesthetic "
    "appreciation. Judge it on its own merits the way a typical viewer would, "
    "without adopting any particular person's perspective or taste. Tastes differ "
    "widely and not every artwork deserves a good score; be honest when a piece is "
    "mediocre or unappealing. Do not mention that you are an AI."
)


def build_lapis_question(dims: list[ScoreDimension], persona_blind: bool = False) -> str:
    """Build the scoring question that asks for exactly the active axes.

    Each axis contributes one bullet (its gloss + its integer range) and one
    key in the required JSON object. The LAPIS slider is 0-100 in whole
    numbers, so the prompt asks for integers.

    ``persona_blind`` drops the "as this exact person" framing so the wording
    matches the generic system prompt (see LAPIS_GENERIC_SYSTEM_PROMPT).
    """
    bullets = "\n".join(
        f"- \"{d.key}\": {d.prompt}. Answer with a whole number from "
        f"{d.lo:g} to {d.hi:g}."
        for d in dims
    )
    schema = ", ".join(f'"{d.key}": <integer>' for d in dims)
    if persona_blind:
        intro = (
            "Rate this artwork on each of the axes below. Judge each one "
            "honestly on its own merits, and use the full range instead of "
            "defaulting to the middle:"
        )
        comment_hint = "<one short sentence explaining your rating>"
    else:
        intro = (
            "As this participant, rate this artwork on each of the axes below. "
            "Judge each one honestly the way this exact person would, and use the "
            "full range instead of defaulting to the middle:"
        )
        comment_hint = "<one short in-character sentence explaining your rating>"
    return (
        f"{intro}\n"
        f"{bullets}\n\n"
        "Respond with ONLY a single JSON object and nothing else (no markdown, "
        "no extra text), in exactly this form:\n"
        f'{{{schema}, "comment": "{comment_hint}"}}'
    )


ANONYMOUS_DESCRIPTION = (
    "a person who chose not to share any demographic information; nothing is "
    "known about their age, background, or interest in art."
)

# VAIAK art interest is the mean of items rated 0-6; the observed data spans
# ~0.1-5.2. Band it so the model gets a qualitative anchor, not just a number.
def _art_interest_band(value: float) -> str:
    if value <= 2:
        return "low"
    if value <= 4:
        return "moderate"
    return "high"


def build_lapis_description(user: pd.Series) -> str:
    """Turn one participant's demographic row into the free-text description.

    Every field can be missing (36 of 568 participants skipped the survey
    entirely — see ANONYMOUS_DESCRIPTION); a present-but-unusable value such
    as gender 'other/would prefer not to disclose' is simply omitted.
    """
    if user[["age", "nationality", "demo_gender", "demo_edu"]].isna().all():
        return ANONYMOUS_DESCRIPTION

    subject = "a person"
    gender = str(user["demo_gender"]) if pd.notna(user["demo_gender"]) else ""
    if gender in ("female", "male", "non-binary"):
        subject = f"a {gender} person"
    if pd.notna(user["nationality"]):
        subject = subject.replace("person", f"{str(user['nationality']).title()} person")
    if pd.notna(user["age"]):
        subject += f" aged {int(user['age'])}"

    clauses = [subject]
    if pd.notna(user["demo_edu"]):
        clauses.append(f"education level: {user['demo_edu']}")
    if pd.notna(user["Art Interest VAIAK"]):
        vaiak = float(user["Art Interest VAIAK"])
        clauses.append(
            f"interest in art: {_art_interest_band(vaiak)} "
            f"(VAIAK art-interest score {vaiak:.1f} on a 0-6 scale)"
        )
    if pd.notna(user["demo_colorblind"]) and str(user["demo_colorblind"]) != "No":
        clauses.append("is colour-blind, but still perceives colours")

    return "; ".join(clauses) + "."


def lapis_system_prompt(description: str) -> str:
    return LAPIS_SYSTEM_PROMPT_TEMPLATE.format(description=description)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

_DEMO_COLS = [
    "age", "nationality", "demo_gender", "demo_edu", "demo_colorblind",
    "Art Interest VAIAK",
]


def load_lapis_tables(dim_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PIAA ratings and a one-row-per-participant demographics frame.

    LAPIS ships with mixed encodings (artist names are Latin-1), so utf-8 is
    tried first with a latin-1 fallback. The ratings frame is returned with
    PARA-style column names (imageName, userId) so the shared sampling/metrics
    code — which groups by imageName — can be reused as-is; rows whose image
    file is missing on disk (a handful of Latin-1-mangled filenames) are
    dropped up front. Demographics are constant per participant in the data,
    so taking the first row per participant loses nothing.
    """
    path = LAPIS_ANNOTATION_DIR / "LAPIS_PIAA.csv"
    usecols = list(dict.fromkeys(
        ["image_filename", "participant_id", *dim_keys, *_DEMO_COLS]
    ))
    try:
        piaa = pd.read_csv(path, encoding="utf-8", usecols=usecols)
    except UnicodeDecodeError:
        piaa = pd.read_csv(path, encoding="latin-1", usecols=usecols)
    piaa = piaa.rename(columns={"image_filename": "imageName", "participant_id": "userId"})

    on_disk = {p.name for p in LAPIS_IMAGE_DIR.iterdir()}
    missing = ~piaa["imageName"].isin(on_disk)
    if missing.any():
        n_img = piaa.loc[missing, "imageName"].nunique()
        print(
            f"warning: dropping {int(missing.sum())} ratings on {n_img} images "
            "whose file is missing from the images directory"
        )
        piaa = piaa[~missing]

    users = piaa.groupby("userId")[_DEMO_COLS].first()
    votes = piaa[["imageName", "userId", *dim_keys]].reset_index(drop=True)
    return votes, users


# --------------------------------------------------------------------------
# Chunked log summary (LAPIS-specific fields; chunk IO reused from para_pipeline)
# --------------------------------------------------------------------------


def _task_key(record: dict) -> tuple:
    """Identity of a rating: LAPIS has no sessionId, so (image, participant)
    suffices. userId round-trips through JSON as a string, so normalize both
    sides to str to keep resume dedup and merge stable."""
    return (record["imageName"], str(record["userId"]))


def _build_summary(
    args: argparse.Namespace,
    dims: list[ScoreDimension],
    question: str,
    descriptions: dict,
    results: list[dict],
    metrics: dict | None,
    chunk_size: int,
    summary_path: Path,
) -> dict:
    """Build the small summary log: config, personas, metrics, and a manifest of
    the result-chunk files. The ratings live in ``.part-NNNN.json`` next to it,
    so the summary stays openable at full-dataset scale."""
    return {
        "dataset": "LAPIS",
        "image_root": str(LAPIS_IMAGE_DIR),
        "model_name": args.model_name,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling": args.sampling if not args.images else "explicit",
        "shard": args.shard,
        "raters_per_image": args.raters_per_image,
        "include_anonymous": args.include_anonymous,
        "persona_blind": args.persona_blind,
        "dimensions": [d.key for d in dims],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt_template": (
            LAPIS_GENERIC_SYSTEM_PROMPT if args.persona_blind else LAPIS_SYSTEM_PROMPT_TEMPLATE
        ),
        "question": question,
        "users": descriptions,
        "chunk_size": chunk_size,
        "n_ratings": len(results),
        "result_parts": [
            _part_path(summary_path, k).name
            for k in range(1, _n_parts(len(results), chunk_size) + 1)
        ],
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# CLI and orchestration
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LAPIS participants as VLM personas on the artworks they "
            "rated and compare simulated vs real 0-100 aesthetic scores."
        )
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of LAPIS images to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated image_filename values to run instead of sampling "
        "(e.g. 'boris-kustodiev_shells-1918.jpg').",
    )
    parser.add_argument(
        "--sampling",
        choices=["stratified", "uniform"],
        default="stratified",
        help="How to sample images: 'stratified' spreads over the human mean-score "
        "range, 'uniform' is plain random (default: %(default)s).",
    )
    parser.add_argument(
        "--raters-per-image",
        type=int,
        default=None,
        help="Cap on how many of an image's real participants to simulate "
        "(default: all of them, ~24 — needed for distribution matching).",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated LAPIS rating axes to elicit and evaluate "
        f"(default: {','.join(ACTIVE_DIMENSIONS)}). "
        f"Choices: {','.join(LAPIS_DIMENSIONS)}.",
    )
    parser.add_argument(
        "--include-anonymous",
        action="store_true",
        help="Also simulate participants with no demographics, using a generic "
        "persona description (default: their ratings are skipped).",
    )
    parser.add_argument(
        "--persona-blind",
        action="store_true",
        help="Ignore every participant's demographics/art interest: judge each "
        "artwork with one generic prompt (no persona) instead of role-playing the "
        "real participant. A model-side persona-blind baseline. Each real "
        "participant is still run (and compared against their true score), so pair "
        "with --temperature > 0 (e.g. 0.7) or every rater of an image gets an "
        "identical answer and the per-image distribution collapses to a point.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="Run only image shard i of N (0-based), so N processes — each pinned "
        "to a GPU with CUDA_VISIBLE_DEVICES — can cover disjoint images in "
        "parallel. Sharding is by image (all raters of an image stay together, "
        "so per-image distribution metrics stay valid). Each shard's log gets a "
        ".shardIofN suffix.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HF model id for the backend (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        choices=["qwen", "llava"],
        default="qwen",
        help="Which VLM backend family to use (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 means greedy/deterministic (default: 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation budget per rating (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many (image, rater) ratings to run through the model in one "
        "padded generate() call. Higher = faster on a big GPU but more VRAM; "
        "drop to 1 for the old one-at-a-time behaviour (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON log. Defaults to data/logs/lapis_<timestamp>.json.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="LOG_JSON",
        help="Continue a previous (possibly crashed/killed) run: load ratings "
        "already present in this log, skip their tasks, and keep appending to "
        "the same file. Must be run with the same selection args (--n-images/"
        "--images/--shard/--seed/--dimensions/--include-anonymous) as the "
        "original run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=60.0,
        help="Seconds between partial-progress saves to the log while running, so "
        "a kill loses at most this much work (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Max ratings per result-chunk file (<stem>.part-NNNN.json); the "
        "summary <stem>.json stays small and holds config + metrics "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--analyze-only",
        default=None,
        metavar="LOG_JSON[,LOG_JSON...]",
        help="Skip the model entirely: recompute metrics from an existing log and "
        "print the summary. A single log is updated in place; several "
        "comma-separated logs (e.g. per-shard outputs of a multi-GPU run) are "
        "merged into combined metrics written to --output or a lapis_merged_* file.",
    )
    return parser.parse_args()


def _coarsen_for_display(dm: dict, max_bins: int = 25) -> dict:
    """Rebin the pooled histograms for the terminal summary. The 0-100 grid has
    101 points, which makes the shared printer's one-line histograms unreadable;
    the log and metrics keep the full-resolution grid — only the printed copy is
    coarsened. Each displayed value is the bin's lower edge."""
    dist = dm.get("distribution")
    if not dist or len(dist["score_grid"]) <= max_bins:
        return dm
    grid = dist["score_grid"]
    factor = int(np.ceil(len(grid) / max_bins))

    def rebin(hist: list) -> list:
        return [float(sum(hist[i : i + factor])) for i in range(0, len(hist), factor)]

    dm = dict(dm)
    dm["distribution"] = {
        **dist,
        "score_grid": [float(grid[i]) for i in range(0, len(grid), factor)],
        "human_hist": rebin(dist["human_hist"]),
        "vlm_hist": rebin(dist["vlm_hist"]),
    }
    return dm


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("LAPIS fidelity summary")
    print("=" * 72)
    print(
        f"Ratings: {metrics['n_ratings']}   "
        f"axes: {', '.join(metrics['dimensions'])}"
    )
    for key in metrics["dimensions"]:
        dm = dict(metrics["per_dimension"][key])
        dm.setdefault("n_ratings", metrics["n_ratings"])
        _print_dimension_summary(key, _coarsen_for_display(dm))


def analyze_log(log_paths: list[Path], output: Path | None = None) -> None:
    """Recompute metrics from one or more logs.

    A single log is updated in place (unless ``output`` is given). Multiple logs
    — e.g. the per-shard outputs of a multi-GPU run — are merged: their results
    are concatenated (deduplicated by (imageName, userId), first log wins) and
    combined metrics are written to ``output`` or a lapis_merged_* file.
    """
    logs, per_log_results = [], []
    for p in log_paths:
        log, res = _read_log_and_results(p)
        logs.append(log)
        per_log_results.append(res)

    dims = resolve_dimensions(logs[0].get("dimensions") or list(ACTIVE_DIMENSIONS))

    merged: dict[tuple, dict] = {}
    for res in per_log_results:
        for r in res:
            merged.setdefault(_task_key(r), r)
    results = list(merged.values())

    metrics = compute_metrics(results, dims, seed=int(logs[0].get("seed", 0)))
    chunk_size = int(logs[0].get("chunk_size", DEFAULT_CHUNK_SIZE))

    if len(log_paths) == 1 and output is None:
        # In-place: the ratings on disk don't change, only the metrics do.
        target, out = log_paths[0], dict(logs[0])
        out["metrics"] = metrics
        out["n_ratings"] = len(results)
        if "result_parts" in logs[0]:  # chunked: refresh manifest, parts untouched
            out["chunk_size"] = chunk_size
            out["result_parts"] = [
                _part_path(target, k).name
                for k in range(1, _n_parts(len(results), chunk_size) + 1)
            ]
        else:  # legacy single-file log: keep the ratings inline
            out["results"] = results
    else:
        target = output or (
            DEFAULT_LOG_DIR
            / f"lapis_merged_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _flush_chunks(target, results, chunk_size, 0)  # write all merged chunks
        out = dict(logs[0])
        out.pop("results", None)
        out.update(
            chunk_size=chunk_size,
            n_ratings=len(results),
            result_parts=[
                _part_path(target, k).name
                for k in range(1, _n_parts(len(results), chunk_size) + 1)
            ],
            metrics=metrics,
            merged_from=[str(p) for p in log_paths],
            users={k: v for log in logs for k, v in log.get("users", {}).items()},
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, out)
    src = f"{len(log_paths)} logs" if len(log_paths) > 1 else str(log_paths[0])
    print(f"Recomputed metrics from {src} -> {target} ({metrics['n_ratings']} ratings).")
    print_summary(metrics)


def run(args: argparse.Namespace) -> None:
    if args.dimensions:
        dim_keys = [k.strip() for k in args.dimensions.split(",") if k.strip()]
    else:
        dim_keys = list(ACTIVE_DIMENSIONS)
    dims = resolve_dimensions(dim_keys)
    question = build_lapis_question(dims, persona_blind=args.persona_blind)
    if args.persona_blind and args.temperature == 0:
        print(
            "warning: --persona-blind with --temperature 0 gives every rater of an "
            "image the identical response (no distribution). Consider --temperature 0.7."
        )

    votes, users = load_lapis_tables(dim_keys)
    anonymous_ids = set(users.index[users[_DEMO_COLS].isna().all(axis=1)])
    if not args.include_anonymous:
        n_before = len(votes)
        votes = votes[~votes["userId"].isin(anonymous_ids)].reset_index(drop=True)
        print(
            f"Excluded {n_before - len(votes)} ratings from "
            f"{len(anonymous_ids)} anonymous participants "
            "(rerun with --include-anonymous to keep them)."
        )

    if args.images:
        selected = [name.strip() for name in args.images.split(",") if name.strip()]
        known = set(votes["imageName"])
        missing = [name for name in selected if name not in known]
        if missing:
            raise ValueError(f"image_filename(s) not found in LAPIS_PIAA.csv: {missing}")
    else:
        selected = choose_images(
            votes, args.n_images, args.seed, args.sampling, dims[0].key
        )

    if args.shard:
        shard_i, shard_n = _parse_shard(args.shard)
        # Round-robin over the (stratified) order so each shard stays balanced.
        selected = selected[shard_i::shard_n]
        print(
            f"Shard {shard_i}/{shard_n}: this process handles {len(selected)} "
            f"of the selected images."
        )

    rng = np.random.default_rng(args.seed)
    tasks: list[dict] = []
    for name in selected:
        rows = votes[votes["imageName"] == name]
        if args.raters_per_image is not None and len(rows) > args.raters_per_image:
            rows = rows.iloc[
                rng.choice(len(rows), size=args.raters_per_image, replace=False)
            ]
        for row in rows.itertuples(index=False):
            task = {"imageName": row.imageName, "userId": int(row.userId)}
            for d in dims:
                task[f"gt_{d.key}"] = float(getattr(row, d.key))
            tasks.append(task)

    n_users = len({t["userId"] for t in tasks})
    print(
        f"Plan: {len(selected)} images x their real participants "
        f"= {len(tasks)} ratings from {n_users} distinct participants."
    )

    descriptions = {
        user_id: build_lapis_description(users.loc[user_id])
        for user_id in {t["userId"] for t in tasks}
    }

    # Resolve the output path and, when resuming, load what is already on disk
    # and drop the tasks whose ratings are already present. The per-shard suffix
    # keeps parallel shards from clobbering one file.
    results: list[dict] = []
    prior_chunked = False
    if args.resume:
        output_path = Path(args.resume)
        prior_log, results = _read_log_and_results(output_path)
        if prior_log.get("dimensions") not in (None, [d.key for d in dims]):
            raise SystemExit(
                f"--resume log has dimensions {prior_log.get('dimensions')}, but "
                f"this run asks for {[d.key for d in dims]}; resume with matching "
                "selection args."
            )
        if bool(prior_log.get("persona_blind", False)) != bool(args.persona_blind):
            raise SystemExit(
                f"--resume log has persona_blind={prior_log.get('persona_blind', False)}, "
                f"which doesn't match --persona-blind={args.persona_blind}; "
                "persona and persona-blind ratings can't share a log."
            )
        prior_chunked = "result_parts" in prior_log
        # Reuse the prior chunk size so part boundaries stay consistent.
        chunk_size = int(prior_log.get("chunk_size", args.chunk_size))
        descriptions = {**prior_log.get("users", {}), **descriptions}
        done = {_task_key(r) for r in results}
        before = len(tasks)
        tasks = [t for t in tasks if _task_key(t) not in done]
        print(
            f"Resuming from {output_path}: {before - len(tasks)} ratings already "
            f"done, {len(tasks)} remaining."
        )
    else:
        chunk_size = args.chunk_size
        suffix = _shard_suffix(args.shard)
        if args.output:
            base = Path(args.output)
            output_path = base.with_name(base.stem + suffix + base.suffix)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = DEFAULT_LOG_DIR / f"lapis_{timestamp}{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tasks:
        # A fully-completed run being re-resumed: refresh metrics + summary and
        # stop without loading the model.
        print("All tasks already present in the resume log; recomputing metrics.")
        metrics = compute_metrics(results, dims, seed=args.seed)
        print_summary(metrics)
        _flush_chunks(output_path, results, chunk_size, 0)
        _atomic_write_json(
            output_path,
            _build_summary(args, dims, question, descriptions, results, metrics, chunk_size, output_path),
        )
        print(f"\nWrote log to {output_path}")
        return

    print(f"Loading {args.backend} backend '{args.model_name}'...")
    from persona import LlavaBackend, QwenVLBackend  # deferred: pulls in torch

    backend_cls = QwenVLBackend if args.backend == "qwen" else LlavaBackend
    backend = backend_cls(model_name=args.model_name)

    gen_kwargs: dict = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        gen_kwargs["do_sample"] = False

    def _run_batch(batch: list[dict]) -> list[str]:
        """Generate for a batch, falling back to one-at-a-time if the fused call
        blows up (a single unreadable image or a transient OOM shouldn't cost the
        whole batch)."""
        images = [LAPIS_IMAGE_DIR / t["imageName"] for t in batch]
        if args.persona_blind:
            sys_prompts = [LAPIS_GENERIC_SYSTEM_PROMPT] * len(batch)
        else:
            sys_prompts = [lapis_system_prompt(descriptions[t["userId"]]) for t in batch]
        try:
            return backend.generate_batch(
                system_prompts=sys_prompts,
                images=images,
                prompts=[question] * len(batch),
                **gen_kwargs,
            )
        except Exception as exc:
            print(f"  batch failed ({exc}); retrying items individually", flush=True)
            out = []
            for sp, im in zip(sys_prompts, images):
                try:
                    out.append(backend.generate(sp, im, question, **gen_kwargs))
                except Exception as inner:
                    out.append(f"<generation error: {inner}>")
            return out

    total = len(tasks)
    n_before_run = len(results)
    # Ratings loaded from an already-chunked resume are on disk; a legacy inline
    # resume (flushed=0) gets rewritten into chunks on the first flush.
    flushed = len(results) if prior_chunked else 0
    started = time.time()
    last_checkpoint = started
    try:
        for start in range(0, total, args.batch_size):
            batch = tasks[start : start + args.batch_size]
            call_start = time.time()
            raw_responses = _run_batch(batch)
            for task, raw_response in zip(batch, raw_responses):
                pred_scores, comment = parse_rating(raw_response, dims)
                record = {**task, "comment": comment, "raw_response": raw_response}
                for d in dims:
                    record[f"pred_{d.key}"] = pred_scores[d.key]
                results.append(record)

            done = start + len(batch)
            batch_dt = time.time() - call_start
            rate = done / max(time.time() - started, 1e-9)
            eta_min = (total - done) / rate / 60 if rate else float("nan")
            print(
                f"[{done:>5}/{total}] batch {len(batch)} in {batch_dt:.1f}s "
                f"({batch_dt / len(batch):.2f}s/item, {rate:.2f} item/s, "
                f"ETA {eta_min:.1f} min)",
                flush=True,  # keep progress visible when stdout is a file/pipe
            )
            if time.time() - last_checkpoint >= args.checkpoint_interval:
                # Partial checkpoint: flush only the tail chunk(s) written since
                # the last save, then the (metrics-free) summary. Cheap even at
                # full-dataset scale, and a kill loses at most one interval.
                flushed = _flush_chunks(output_path, results, chunk_size, flushed)
                _atomic_write_json(
                    output_path,
                    _build_summary(args, dims, question, descriptions, results, None, chunk_size, output_path),
                )
                last_checkpoint = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(results) - n_before_run}/{total} new "
              "ratings this run — writing partial log.")

    elapsed = time.time() - started
    print(f"\nGenerated {len(results) - n_before_run} ratings in {elapsed / 60:.1f} min "
          f"({len(results)} total in log).")

    metrics = compute_metrics(results, dims, seed=args.seed)
    print_summary(metrics)

    flushed = _flush_chunks(output_path, results, chunk_size, flushed)
    _atomic_write_json(
        output_path,
        _build_summary(args, dims, question, descriptions, results, metrics, chunk_size, output_path),
    )
    print(f"\nWrote log to {output_path}")


def main() -> None:
    args = _parse_args()
    if args.analyze_only:
        paths = [Path(p.strip()) for p in args.analyze_only.split(",") if p.strip()]
        analyze_log(paths, Path(args.output) if args.output else None)
        return
    run(args)


if __name__ == "__main__":
    main()
