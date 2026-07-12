"""EVA fidelity pipeline: replay real EVA voters as VLM personas.

The EVA dataset (data/eva-dataset/) contains ~137k filtered (image, voter)
aesthetic ballots on ~4k images resized from AVA. Every ballot carries an
overall 0-10 score plus four ordinal 1-4 attributes (visual, composition,
quality, semantic) and a 1-4 difficulty; every voter has demographics
(birth year, region, gender, photographic expertise, eyesight).

For each sampled image this script re-creates *the exact voters who rated it*
(~30 per image) as VLM personas and asks each one, in character, for the same
ratings the study collected. Fidelity is then measured at two levels:

1. per-rating — does the simulated person give the same score as the real
   person on the same image? (MAE / RMSE / correlation, against baselines)
2. per-image  — do the ~30 simulated voters reproduce the *score
   distribution* of the 30 human voters? (EMD / KS vs a bootstrap noise floor)

Per the repo convention, EVA-specific prompts (persona-description builder,
system prompt, scoring question, score axes) live here, not in src/persona:
EVA voters have different features than PARA annotators (no Big Five; region
and eyesight instead) and a different scale (0-10 integers vs 1-5 halves).
The dataset-agnostic machinery (ScoreDimension grids, JSON parsing, metrics,
stratified sampling) is imported from the sibling script/para_pipeline.py.

Run from the repo root (persona conda env):

    python script/eva_pipeline.py --n-images 5 --seed 0
    python script/eva_pipeline.py --images 71,106 --raters-per-image 3
    python script/eva_pipeline.py --dimensions score,visual,composition
    python script/eva_pipeline.py --analyze-only data/logs/eva_<ts>.json

A full-dataset run is one generate() call per ballot (~137k), so fan it out
across GPUs with --shard i/N: each of N processes takes a disjoint slice of the
images (all voters of an image stay in one shard, keeping per-image
distribution metrics valid) and writes its own .shardIofN log. E.g. over 4 GPUs:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python script/eva_pipeline.py \
        --n-images 4070 --shard $i/4 --output data/logs/eva_full.json &
    done

yields data/logs/eva_full.shard0of4.json,...,eva_full.shard3of4.json, which
--analyze-only reads together for combined metrics.

Each run writes a small summary log (config, personas, metrics, and a manifest
of its result-chunk files) plus the ratings themselves in fixed-size
<stem>.part-NNNN.json chunks. The log is checkpointed *while running* (every
--checkpoint-interval seconds, default 60), so a crash or kill loses at most one
interval; resume where it stopped with:

    python script/eva_pipeline.py ... --resume data/logs/eva_<ts>.json

--resume must be paired with the same selection args (--n-images/--images/
--shard/--seed/--dimensions) as the original run; the (imageName, userId)
ratings already in the log are skipped and only the rest are generated.

--analyze-only recomputes metrics from such a log (or several) without touching
the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make ``src`` (persona package) and ``script`` (para_pipeline) importable when
# this file is run from anywhere, not just via ``python script/eva_pipeline.py``.
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

EVA_DATA_DIR = REPO_ROOT / "data" / "eva-dataset" / "data"
EVA_IMAGE_DIR = REPO_ROOT / "data" / "eva-dataset" / "images" / "EVA_together"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

# --------------------------------------------------------------------------
# EVA score axes (columns of votes_filtered.csv)
# --------------------------------------------------------------------------
# The overall score is an integer 0-10; the four aesthetic attributes and the
# judging difficulty are ordinal 1-4. All grids are integer-stepped.

EVA_DIMENSIONS: dict[str, ScoreDimension] = {
    "score": ScoreDimension(
        "score", 0.0, 10.0, 1.0,
        "overall aesthetic score (0 = very bad, 5 = average, 10 = excellent)",
    ),
    "visual": ScoreDimension(
        "visual", 1.0, 4.0, 1.0,
        "light and colour (1 = very bad, 2 = bad, 3 = good, 4 = very good)",
    ),
    "composition": ScoreDimension(
        "composition", 1.0, 4.0, 1.0,
        "composition and depth (1 = very bad, 2 = bad, 3 = good, 4 = very good)",
    ),
    "quality": ScoreDimension(
        "quality", 1.0, 4.0, 1.0,
        "technical quality — sharpness, exposure, noise "
        "(1 = very bad, 2 = bad, 3 = good, 4 = very good)",
    ),
    "semantic": ScoreDimension(
        "semantic", 1.0, 4.0, 1.0,
        "semantic content — how interesting or meaningful the subject is "
        "(1 = very bad, 2 = bad, 3 = good, 4 = very good)",
    ),
    "difficulty": ScoreDimension(
        "difficulty", 1.0, 4.0, 1.0,
        "how easy this image was to judge "
        "(1 = very difficult, 2 = difficult, 3 = easy, 4 = very easy)",
    ),
}

# Which axes to elicit and evaluate by default; the first entry is the primary
# axis used for image stratification and the progress readout. Override at
# runtime with --dimensions.
ACTIVE_DIMENSIONS = list(EVA_DIMENSIONS)


def resolve_dimensions(keys: list[str]) -> list[ScoreDimension]:
    """Map a list of EVA axis keys to ScoreDimension objects, validating each."""
    if not keys:
        raise ValueError("no score dimensions selected")
    unknown = [k for k in keys if k not in EVA_DIMENSIONS]
    if unknown:
        raise ValueError(
            f"unknown score dimension(s): {unknown}; "
            f"choose from {list(EVA_DIMENSIONS)}"
        )
    return [EVA_DIMENSIONS[k] for k in keys]


# --------------------------------------------------------------------------
# EVA-specific prompts
# --------------------------------------------------------------------------

EVA_SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing as one specific human participant in an online study "
    "of visual aesthetics. The participant is {description}\n\n"
    "You are shown one photograph from the study. Judge it the way this exact "
    "person would: let their age, gender, cultural background, photography "
    "experience, and eyesight shape both what they notice and how critical or "
    "generous they are. Not every photo deserves a good score; be honest when "
    "a photo would bore or disappoint this person. Do not mention that you are "
    "an AI or that you are role-playing."
)

# Persona-blind baseline: judge the image with no rater conditioning at all, so
# the model's generic opinion can be contrasted with the persona-conditioned
# runs (and the statistical predict-mean baselines). Selected with --persona-blind.
EVA_GENERIC_SYSTEM_PROMPT = (
    "You are an impartial, general-purpose judge of visual aesthetics.\n\n"
    "You are shown one photograph from a study of visual aesthetics. Judge it on "
    "its own merits the way a typical viewer would, without adopting any "
    "particular person's perspective or taste. Not every photo deserves a good "
    "score; be honest when a photo is mediocre or disappointing. Do not mention "
    "that you are an AI."
)


def build_eva_question(dims: list[ScoreDimension], persona_blind: bool = False) -> str:
    """Build the scoring question that asks for exactly the active axes.

    Each axis contributes one bullet (its gloss + its integer range) and one key
    in the required JSON object, so adding an axis to ACTIVE_DIMENSIONS
    automatically extends both what the model is asked for and the schema it
    must return. All EVA grids are integer-stepped, so the prompt asks for
    whole numbers.

    ``persona_blind`` drops the "as this exact person" framing so the wording
    matches the generic system prompt (see EVA_GENERIC_SYSTEM_PROMPT).
    """
    bullets = "\n".join(
        f"- \"{d.key}\": {d.prompt}. Answer with a whole number from "
        f"{d.lo:g} to {d.hi:g}."
        for d in dims
    )
    schema = ", ".join(f'"{d.key}": <integer>' for d in dims)
    if persona_blind:
        intro = (
            "Rate this photograph on each of the axes below. Judge each one "
            "honestly on its own merits, and use the full range instead of "
            "defaulting to the middle:"
        )
        comment_hint = "<one short sentence explaining your ratings>"
    else:
        intro = (
            "As this participant, rate this photograph on each of the axes below. "
            "Judge each one honestly the way this exact person would, and use the "
            "full range instead of defaulting to the middle:"
        )
        comment_hint = "<one short in-character sentence explaining your ratings>"
    return (
        f"{intro}\n"
        f"{bullets}\n\n"
        "Respond with ONLY a single JSON object and nothing else (no markdown, "
        "no extra text), in exactly this form:\n"
        f'{{{schema}, "comment": "{comment_hint}"}}'
    )


_GENDER = {1: "male", 2: "female"}
_PHOTO_LEVEL = {
    0: "no experience with photography",
    1: "a beginner at photography",
    2: "an amateur photographer",
    3: "a professional photographer",
}
EVA_REF_YEAR = 2020  # experiment year; users.csv 'age' column is a birth year


def build_eva_description(user: pd.Series, region_names: dict[int, str]) -> str:
    """Turn one users.csv row into the free-text persona description.

    Handles missing fields by omitting them: every demographic is optional
    except that at least a generic 'person' is always described. eyecheck is
    blank for most voters (= no glasses, not colour-blind) and is only
    mentioned when set, since colour-blindness in particular should change how
    a rater sees an image.
    """
    gender = _GENDER.get(int(user["gender_id"]) if pd.notna(user["gender_id"]) else -1)
    subject = f"a {gender} person" if gender else "a person"
    if pd.notna(user["age"]) and 5 <= EVA_REF_YEAR - int(user["age"]) <= 100:
        subject += f" aged {EVA_REF_YEAR - int(user['age'])}"
    if pd.notna(user["region"]):
        region = region_names.get(int(user["region"]))
        if region:
            subject += f" from {region}"

    clauses = [subject]
    if pd.notna(user["photographic_level_id"]):
        level = _PHOTO_LEVEL.get(int(user["photographic_level_id"]))
        if level:
            clauses.append(f"has {level}" if level.startswith("no ") else f"is {level}")

    eyecheck = str(user["eyecheck"]) if pd.notna(user["eyecheck"]) else ""
    if "1" in eyecheck.split(","):
        clauses.append("wears corrective glasses")
    if "2" in eyecheck.split(","):
        clauses.append("is colour-blind")

    return "; ".join(clauses) + "."


def eva_system_prompt(description: str) -> str:
    return EVA_SYSTEM_PROMPT_TEMPLATE.format(description=description)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_eva_tables(dim_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Load votes_filtered (only the needed columns), users, and region names.

    The votes frame is returned with PARA-style column names (imageName,
    userId) so the shared sampling/metrics code — which groups by imageName —
    can be reused as-is. imageName is the AVA image id as a string; the file on
    disk is ``EVA_IMAGE_DIR/<imageName>.jpg``.
    """
    usecols = list(dict.fromkeys(["image_id", "user_id", *dim_keys]))
    votes = pd.read_csv(EVA_DATA_DIR / "votes_filtered.csv", sep="=", usecols=usecols)
    votes = votes.rename(columns={"image_id": "imageName", "user_id": "userId"})
    votes["imageName"] = votes["imageName"].astype(str)

    users = pd.read_csv(EVA_DATA_DIR / "users.csv", sep="=").set_index("id")
    regions = pd.read_csv(
        EVA_DATA_DIR / "region_index.csv", sep="=", header=None,
        names=["region_code", "region_name"],
    )
    region_names = dict(zip(regions["region_code"], regions["region_name"]))
    return votes, users, region_names


# --------------------------------------------------------------------------
# Chunked log summary (EVA-specific fields; chunk IO reused from para_pipeline)
# --------------------------------------------------------------------------


def _task_key(record: dict) -> tuple:
    """Identity of a rating: EVA has no sessionId, so (image, voter) suffices."""
    return (record["imageName"], record["userId"])


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
        "dataset": "EVA",
        "image_root": str(EVA_IMAGE_DIR),
        "model_name": args.model_name,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling": args.sampling if not args.images else "explicit",
        "shard": args.shard,
        "raters_per_image": args.raters_per_image,
        "persona_blind": args.persona_blind,
        "dimensions": [d.key for d in dims],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt_template": (
            EVA_GENERIC_SYSTEM_PROMPT if args.persona_blind else EVA_SYSTEM_PROMPT_TEMPLATE
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
            "Replay EVA voters as VLM personas on the images they rated and "
            "compare simulated vs real 0-10 aesthetic scores (and optionally "
            "the 1-4 attribute ratings)."
        )
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of EVA images to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated image_id values to run instead of sampling "
        "(e.g. '71,106' — AVA image names without .jpg).",
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
        help="Cap on how many of an image's real voters to simulate "
        "(default: all of them, ~30 — needed for distribution matching).",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated EVA rating axes to elicit and evaluate "
        f"(default: {','.join(ACTIVE_DIMENSIONS)}). "
        f"Choices: {','.join(EVA_DIMENSIONS)}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    parser.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="Run only image shard i of N (0-based), so N processes — each pinned "
        "to a GPU with CUDA_VISIBLE_DEVICES — can cover disjoint images in "
        "parallel. Sharding is by image (all voters of an image stay together, "
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
        "--persona-blind",
        action="store_true",
        help="Ignore every voter's demographics: judge each image with one "
        "generic prompt (no persona) instead of role-playing the real voter. A "
        "model-side persona-blind baseline. Each real voter is still run (and "
        "compared against their true score), so pair with --temperature > 0 "
        "(e.g. 0.7) or every voter of an image gets an identical answer and the "
        "per-image distribution collapses to a point.",
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
        help="Where to write the JSON log. Defaults to data/logs/eva_<timestamp>.json.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="LOG_JSON",
        help="Continue a previous (possibly crashed/killed) run: load ratings "
        "already present in this log, skip their tasks, and keep appending to "
        "the same file. Must be run with the same selection args (--n-images/"
        "--images/--shard/--seed/--dimensions) as the original run.",
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
        "merged into combined metrics written to --output or an eva_merged_* file.",
    )
    return parser.parse_args()


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("EVA fidelity summary")
    print("=" * 72)
    print(
        f"Ratings: {metrics['n_ratings']}   "
        f"axes: {', '.join(metrics['dimensions'])}"
    )
    for key in metrics["dimensions"]:
        dm = dict(metrics["per_dimension"][key])
        dm.setdefault("n_ratings", metrics["n_ratings"])
        _print_dimension_summary(key, dm)


def analyze_log(log_paths: list[Path], output: Path | None = None) -> None:
    """Recompute metrics from one or more logs.

    A single log is updated in place (unless ``output`` is given). Multiple logs
    — e.g. the per-shard outputs of a multi-GPU run — are merged: their results
    are concatenated (deduplicated by (imageName, userId), first log wins) and
    combined metrics are written to ``output`` or an eva_merged_* file.
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
            / f"eva_merged_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
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
    import numpy as np

    if args.dimensions:
        dim_keys = [k.strip() for k in args.dimensions.split(",") if k.strip()]
    else:
        dim_keys = list(ACTIVE_DIMENSIONS)
    dims = resolve_dimensions(dim_keys)
    question = build_eva_question(dims, persona_blind=args.persona_blind)
    if args.persona_blind and args.temperature == 0:
        print(
            "warning: --persona-blind with --temperature 0 gives every voter of an "
            "image the identical response (no distribution). Consider --temperature 0.7."
        )

    votes, users, region_names = load_eva_tables(dim_keys)

    if args.images:
        selected = [name.strip() for name in args.images.split(",") if name.strip()]
        known = set(votes["imageName"])
        missing = [name for name in selected if name not in known]
        if missing:
            raise ValueError(f"image_id(s) not found in votes_filtered.csv: {missing}")
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
            if row.userId not in users.index:
                print(f"warning: {row.userId} missing from users.csv, skipped")
                continue
            task = {"imageName": row.imageName, "userId": row.userId}
            for d in dims:
                task[f"gt_{d.key}"] = float(getattr(row, d.key))
            tasks.append(task)

    n_users = len({t["userId"] for t in tasks})
    print(
        f"Plan: {len(selected)} images x their real voters "
        f"= {len(tasks)} ratings from {n_users} distinct voters."
    )

    descriptions = {
        user_id: build_eva_description(users.loc[user_id], region_names)
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
            output_path = DEFAULT_LOG_DIR / f"eva_{timestamp}{suffix}.json"
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
        images = [EVA_IMAGE_DIR / f"{t['imageName']}.jpg" for t in batch]
        if args.persona_blind:
            sys_prompts = [EVA_GENERIC_SYSTEM_PROMPT] * len(batch)
        else:
            sys_prompts = [eva_system_prompt(descriptions[t["userId"]]) for t in batch]
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
                f"[{done:>4}/{total}] batch {len(batch)} in {batch_dt:.1f}s "
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
        print(f"\nInterrupted after {len(results)} ratings — writing partial log.")

    elapsed = time.time() - started
    print(f"\nGenerated {len(results)} ratings total in {elapsed / 60:.1f} min.")

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
