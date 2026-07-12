"""PARA fidelity pipeline: replay real PARA annotators as VLM personas.

The PARA dataset (data/PARA/) contains ~808k (image, annotator) aesthetic
ratings on a 1-5 scale (0.5 steps), where every annotator has a known
demographic + Big-Five profile (data/PARA/annotation/PARA-UserInfo.csv).

For each sampled image this script re-creates *the exact annotators who rated
it* as VLM personas and asks each one, in character, for the same 1-5
aesthetic score the study collected. It then measures fidelity at two levels:

1. per-rating  — does the simulated person give the same score as the real
   person on the same image? (MAE / RMSE / correlation, against baselines)
2. per-image   — does the set of ~25 simulated raters reproduce the *score
   distribution* of the 25 human raters? (EMD / KS vs a bootstrap noise floor)

PARA-specific prompts live here, not in src/persona: each dataset has its own
rater features and its own scale, so each dataset script owns its own system
prompt template, persona-description builder, and scoring question (LAPIS/EVA
should get sibling scripts). The generic src/ pipeline keeps its social-feed
0-100 prompts.

Ratings are generated in batches (--batch-size, one fused model.generate call
per batch) rather than one persona at a time, which is the main throughput lever
on a GPU.

Multi-GPU: this is embarrassingly parallel across images, so to use N GPUs just
launch N processes, each pinned to one GPU with CUDA_VISIBLE_DEVICES and given a
disjoint slice of the images via --shard i/N. Sharding is at the image level, so
every rater of an image stays in the same shard and per-image distribution
metrics remain valid. Each shard writes its own log (a .shardIofN suffix is added
automatically); merge them for combined metrics with a comma-separated
--analyze-only. Example, 4 GPUs over the whole dataset:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python script/para_pipeline.py \
        --n-images 31220 --shard $i/4 --output data/logs/para_full.json &
    done; wait
    python script/para_pipeline.py --analyze-only \
      data/logs/para_full.shard0of4.json,...,para_full.shard3of4.json

Run from the repo root (persona conda env):

    python script/para_pipeline.py --n-images 5 --seed 0 --batch-size 8
    python script/para_pipeline.py --images iaa_pub10_.jpg --raters-per-image 3
    python script/para_pipeline.py --analyze-only data/logs/para_<ts>.json

Output is split so no single file grows unwieldy at full-dataset scale: a small
summary data/logs/para_<timestamp>.json holds the config, personas and metrics
plus a manifest of the result-chunk files, and the ratings themselves (with raw
model responses) go to data/logs/para_<timestamp>.part-NNNN.json, at most
--chunk-size ratings each. --analyze-only reads the chunks back to recompute
metrics without touching the model.

The log is also checkpointed *while the run is in progress* (every
--checkpoint-interval seconds, default 60), so a crash or kill doesn't lose
completed ratings. To continue a run that stopped partway through:

    python script/para_pipeline.py --n-images 5 --seed 0 \
        --resume data/logs/para_<ts>.json

--resume must be paired with the same selection args (--n-images/--images/
--raters-per-image/--seed/--sampling/--dimensions) as the original run, since
that's what makes the task list reproducible; already-completed (sessionId,
imageName, userId) ratings found in the resumed log are skipped and new ones
are appended to that same file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make ``src`` importable when running this file directly (same trick as opinions.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

PARA_ANNOTATION_DIR = REPO_ROOT / "data" / "PARA" / "annotation"
PARA_IMAGE_DIR = REPO_ROOT / "data" / "PARA" / "imgs"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

# A run's ratings are written to fixed-size chunk files (<stem>.part-NNNN.json)
# alongside a small summary <stem>.json, so no single file grows unwieldy at
# full-dataset scale and checkpoints only rewrite the tail chunk.
DEFAULT_CHUNK_SIZE = 5000

# --------------------------------------------------------------------------
# PARA score axes
# --------------------------------------------------------------------------
# PARA-Images.csv carries several per-(image, annotator) rating columns, each on
# its own 1-5 grid: aestheticScore in 0.5 steps, qualityScore in 0.1 steps, and
# the remaining sub-scores as integers. A ScoreDimension bundles a column's grid
# with the natural-language gloss used to ask the model for it, so the prompt,
# parser, and metrics can all be driven off the same registry.


@dataclass(frozen=True)
class ScoreDimension:
    key: str  # the PARA-Images.csv column name and the JSON key we elicit
    lo: float
    hi: float
    step: float
    prompt: str  # natural-language description of the axis and its endpoints

    @property
    def grid(self) -> np.ndarray:
        """Inclusive value grid ``lo, lo+step, ..., hi`` this axis lives on."""
        n = int(round((self.hi - self.lo) / self.step)) + 1
        return np.round(self.lo + self.step * np.arange(n), 6)

    def snap(self, value: float) -> float:
        """Clamp to ``[lo, hi]`` and snap to the nearest grid step (half-up)."""
        steps = np.floor((float(value) - self.lo) / self.step + 0.5)
        snapped = self.lo + steps * self.step
        return float(min(self.hi, max(self.lo, snapped)))


# Every PARA rating axis we know how to elicit. Keys match PARA-Images.csv columns.
PARA_DIMENSIONS: dict[str, ScoreDimension] = {
    "aestheticScore": ScoreDimension(
        "aestheticScore", 1.0, 5.0, 0.5,
        "overall aesthetic appeal (1 = very poor, 3 = average, 5 = excellent)",
    ),
    "qualityScore": ScoreDimension(
        "qualityScore", 1.0, 5.0, 0.1,
        "technical image quality — sharpness, exposure, noise and artifacts "
        "(1 = very low, 5 = flawless)",
    ),
    "compositionScore": ScoreDimension(
        "compositionScore", 1.0, 5.0, 1.0,
        "composition — framing, balance and use of space (1 = poor, 5 = excellent)",
    ),
    "colorScore": ScoreDimension(
        "colorScore", 1.0, 5.0, 1.0,
        "colour — palette, harmony and vibrancy (1 = poor, 5 = excellent)",
    ),
    "dofScore": ScoreDimension(
        "dofScore", 1.0, 5.0, 1.0,
        "depth of field and focus handling (1 = poor, 5 = excellent)",
    ),
    "contentScore": ScoreDimension(
        "contentScore", 1.0, 5.0, 1.0,
        "how interesting or compelling the subject/content is (1 = poor, 5 = excellent)",
    ),
    "lightScore": ScoreDimension(
        "lightScore", 1.0, 5.0, 1.0,
        "lighting quality (1 = poor, 5 = excellent)",
    ),
    "contentPreference": ScoreDimension(
        "contentPreference", 1.0, 5.0, 1.0,
        "how much this person personally likes this kind of content, regardless of "
        "photographic quality (1 = strongly dislike, 5 = strongly like)",
    ),
    "willingnessToShare": ScoreDimension(
        "willingnessToShare", 1.0, 5.0, 1.0,
        "how willing this person would be to share this photo on their own social "
        "media (1 = never, 5 = definitely)",
    ),
}

# Which axes to actually elicit from the model and evaluate. Edit this list to add
# or remove axes (every entry must be a key of PARA_DIMENSIONS); the first entry is
# the "primary" axis used for image stratification and the progress readout. The
# --dimensions CLI flag overrides this at runtime.
ACTIVE_DIMENSIONS = list(PARA_DIMENSIONS)


def resolve_dimensions(keys: list[str]) -> list[ScoreDimension]:
    """Map a list of axis keys to ScoreDimension objects, validating each."""
    if not keys:
        raise ValueError("no score dimensions selected")
    unknown = [k for k in keys if k not in PARA_DIMENSIONS]
    if unknown:
        raise ValueError(
            f"unknown score dimension(s): {unknown}; "
            f"choose from {list(PARA_DIMENSIONS)}"
        )
    return [PARA_DIMENSIONS[k] for k in keys]


# --------------------------------------------------------------------------
# PARA-specific prompts
# --------------------------------------------------------------------------

PARA_SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing as one specific human annotator from a photo-aesthetics "
    "rating study. The annotator is {description}\n\n"
    "You are shown one photograph from the study. Judge it the way this exact "
    "person would: let their age, gender, education, art and photography "
    "experience, and Big-Five personality shape both what they notice and how "
    "critical or generous they are. Not every photo deserves a good score; be "
    "honest when a photo would bore or disappoint this person. Do not mention "
    "that you are an AI or that you are role-playing."
)

# Persona-blind baseline: judge the image with no rater conditioning at all, so
# the model's generic opinion can be contrasted with the persona-conditioned
# runs (and the statistical predict-mean baselines). Selected with --persona-blind.
PARA_GENERIC_SYSTEM_PROMPT = (
    "You are an impartial, general-purpose judge of photographic aesthetics.\n\n"
    "You are shown one photograph from a photo-aesthetics rating study. Judge it "
    "on its own merits the way a typical viewer would, without adopting any "
    "particular person's perspective or taste. Not every photo deserves a good "
    "score; be honest when a photo is mediocre or disappointing. Do not mention "
    "that you are an AI."
)

def _fmt_step(step: float) -> str:
    """'0.5' / '0.1' / '1' — trim trailing zeros for the prompt text."""
    return f"{step:g}"


def build_para_question(dims: list[ScoreDimension], persona_blind: bool = False) -> str:
    """Build the scoring question that asks for exactly the active axes.

    Each axis contributes one bullet (its gloss + its 1-5 grid) and one key in
    the required JSON object, so adding an axis to ACTIVE_DIMENSIONS automatically
    extends both what the model is asked for and the schema it must return.

    ``persona_blind`` drops the "as this exact person" framing so the wording
    matches the generic system prompt (see PARA_GENERIC_SYSTEM_PROMPT).
    """
    bullets = "\n".join(
        f"- \"{d.key}\": {d.prompt}. Rate from {d.lo:g} to {d.hi:g} in steps of "
        f"{_fmt_step(d.step)}."
        for d in dims
    )
    schema = ", ".join(f'"{d.key}": <number>' for d in dims)
    if persona_blind:
        intro = (
            "Rate this photograph on each of the axes below. Judge each one "
            "honestly on its own merits, and use the full range instead of "
            "defaulting to the middle:"
        )
        comment_hint = "<one short sentence explaining your scores>"
    else:
        intro = (
            "As this annotator, rate this photograph on each of the axes below. "
            "Judge each one honestly the way this exact person would, and use the "
            "full range instead of defaulting to the middle:"
        )
        comment_hint = "<one short in-character sentence explaining your scores>"
    return (
        f"{intro}\n"
        f"{bullets}\n\n"
        "Respond with ONLY a single JSON object and nothing else (no markdown, no "
        "extra text), in exactly this form:\n"
        f'{{{schema}, "comment": "{comment_hint}"}}'
    )

_EDU_LABELS = {
    "junior_high_school": "junior high school",
    "senior_high_school": "senior high school",
    "technical_secondary_school": "technical secondary school",
    "junior_college": "junior college",
    "university": "university",
}

_TRAITS = [
    ("personality-E", "extraversion"),
    ("personality-A", "agreeableness"),
    ("personality-N", "neuroticism"),
    ("personality-O", "openness to experience"),
    ("personality-C", "conscientiousness"),
]


def _trait_band(value: float) -> str:
    if value <= 4:
        return "low"
    if value <= 7:
        return "moderate"
    return "high"


def build_para_description(user: pd.Series) -> str:
    """Turn one PARA-UserInfo row into the free-text persona description."""
    edu = _EDU_LABELS.get(
        user["EducationalLevel"], str(user["EducationalLevel"]).replace("_", " ")
    )
    traits = ", ".join(
        f"{name} {int(user[col])}/10 ({_trait_band(user[col])})" for col, name in _TRAITS
    )
    return (
        f"a {user['gender']} aged {user['age']}, education level: {edu}; "
        f"art experience: {user['artExperience']}; "
        f"photography experience: {user['photographyExperience']}. "
        f"Big-Five personality scores (2-10 scale): {traits}."
    )


def para_system_prompt(description: str) -> str:
    return PARA_SYSTEM_PROMPT_TEMPLATE.format(description=description)


# --------------------------------------------------------------------------
# Response parsing (PARA scale: 1-5, per-axis grid; not 0-100 ints)
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _extract_score(data: Optional[dict], text: str, dim: ScoreDimension) -> Optional[float]:
    """Pull one axis's score, snapped to its grid, from parsed JSON or raw text."""
    if isinstance(data, dict) and dim.key in data:
        try:
            return dim.snap(float(data[dim.key]))
        except (TypeError, ValueError):
            pass
    match = re.search(rf'"{re.escape(dim.key)}"\s*:\s*"?(-?\d+(?:\.\d+)?)', text)
    return dim.snap(float(match.group(1))) if match else None


def parse_para_rating(
    raw_response: str, dims: list[ScoreDimension]
) -> tuple[dict[str, Optional[float]], str]:
    """Parse a persona's raw response into ``({axis_key: score_or_None}, comment)``.

    Mirrors src/pipeline.py's parse_rating (strict JSON first, then a per-key
    regex fallback for fenced/dirty output) but over the active PARA axes. A
    missing/unparseable axis yields ``None`` for that axis rather than raising, so
    one malformed field doesn't discard the whole response.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()

    start, end = text.find("{"), text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end > start else text
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        data = None

    scores = {dim.key: _extract_score(data, text, dim) for dim in dims}

    if isinstance(data, dict) and "comment" in data:
        comment = str(data.get("comment", "")).strip()
    else:
        comment_match = _COMMENT_RE.search(text)
        comment = comment_match.group(1) if comment_match else text
    return scores, comment


# --------------------------------------------------------------------------
# Metrics (numpy/pandas only, no scipy in the persona env)
# --------------------------------------------------------------------------


def _hist_on_grid(scores, grid: np.ndarray) -> np.ndarray:
    """Normalized histogram of scores over a value grid."""
    grid = np.asarray(grid, dtype=float)
    step = grid[1] - grid[0] if len(grid) > 1 else 1.0
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros(len(grid))
    # Snap each score to its nearest grid bin in one vectorized pass (the bootstrap
    # noise floor calls this millions of times over the full dataset).
    idx = np.clip(np.rint((scores - grid[0]) / step).astype(int), 0, len(grid) - 1)
    hist = np.bincount(idx, minlength=len(grid)).astype(float)
    total = hist.sum()
    return hist / total if total else hist


def emd(scores_a, scores_b, grid: np.ndarray) -> float:
    """1-D Wasserstein distance between two score samples, in score points."""
    step = grid[1] - grid[0] if len(grid) > 1 else 1.0
    cdf_a = np.cumsum(_hist_on_grid(scores_a, grid))
    cdf_b = np.cumsum(_hist_on_grid(scores_b, grid))
    return float(np.sum(np.abs(cdf_a - cdf_b)) * step)  # step = grid spacing


def ks_stat(scores_a, scores_b, grid: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (max CDF gap) between two score samples."""
    cdf_a = np.cumsum(_hist_on_grid(scores_a, grid))
    cdf_b = np.cumsum(_hist_on_grid(scores_b, grid))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    return _pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def human_resample_emd(gt_scores, n_draw: int, rng, grid: np.ndarray, n_boot: int = 200) -> float:
    """Sampling-noise floor for the per-image EMD: the expected EMD between the
    human ratings and an equally sized bootstrap resample of themselves. A model
    EMD close to this floor is indistinguishable from human sampling noise."""
    gt = np.asarray(gt_scores, dtype=float)
    return float(
        np.mean(
            [emd(rng.choice(gt, size=n_draw, replace=True), gt, grid) for _ in range(n_boot)]
        )
    )


def _dimension_metrics(df: pd.DataFrame, dim: ScoreDimension, rng) -> dict:
    """Fidelity metrics for one axis, keyed off its ``gt_<key>``/``pred_<key>`` columns."""
    gt_col, pred_col = f"gt_{dim.key}", f"pred_{dim.key}"
    grid = dim.grid
    out: dict = {"n_parsed": 0}
    if len(df) == 0 or pred_col not in df.columns:
        return out
    parsed = df[df[pred_col].notna()].copy()
    if len(parsed) == 0:
        return out
    out["n_parsed"] = int(len(parsed))

    gt = parsed[gt_col].to_numpy(dtype=float)
    pred = parsed[pred_col].to_numpy(dtype=float)
    err = pred - gt
    image_mean = parsed.groupby("imageName")[gt_col].transform("mean").to_numpy(dtype=float)

    out["per_rating"] = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mean_bias": float(err.mean()),  # positive = model scores higher than humans
        "pearson": _pearson(gt, pred),
        "spearman": _spearman(gt, pred),
        # what MAE a persona-blind predictor would get:
        "baseline_mae_global_mean": float(np.abs(gt - gt.mean()).mean()),
        "baseline_mae_image_mean": float(np.abs(gt - image_mean).mean()),
    }

    per_image = []
    for name, grp in parsed.groupby("imageName", sort=False):
        g_gt = grp[gt_col].to_numpy(dtype=float)
        g_pred = grp[pred_col].to_numpy(dtype=float)
        per_image.append(
            {
                "imageName": name,
                "n": int(len(grp)),
                "human_mean": float(g_gt.mean()),
                "human_std": float(g_gt.std(ddof=1)) if len(g_gt) > 1 else 0.0,
                "vlm_mean": float(g_pred.mean()),
                "vlm_std": float(g_pred.std(ddof=1)) if len(g_pred) > 1 else 0.0,
                "mae": float(np.abs(g_pred - g_gt).mean()),
                "emd": emd(g_pred, g_gt, grid),
                "ks": ks_stat(g_pred, g_gt, grid),
                "human_resample_emd": human_resample_emd(g_gt, len(g_pred), rng, grid),
            }
        )
    out["per_image"] = per_image

    pi = pd.DataFrame(per_image)
    out["distribution"] = {
        "pooled_emd": emd(pred, gt, grid),
        "pooled_ks": ks_stat(pred, gt, grid),
        "score_grid": grid.tolist(),
        "human_hist": _hist_on_grid(gt, grid).tolist(),
        "vlm_hist": _hist_on_grid(pred, grid).tolist(),
        "image_mean_pearson": _pearson(pi["human_mean"], pi["vlm_mean"]),
        "image_mean_spearman": _spearman(pi["human_mean"], pi["vlm_mean"]),
    }
    return out


def compute_metrics(records: list[dict], dims: list[ScoreDimension], seed: int = 0) -> dict:
    """Fidelity metrics per active axis over per-rating records.

    Each record carries ``gt_<key>``/``pred_<key>`` for every axis in ``dims``;
    metrics are computed independently for each and returned under
    ``per_dimension[key]`` (same structure the single-axis version used to return
    at the top level).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records)
    metrics: dict = {
        "n_ratings": int(len(df)),
        "dimensions": [d.key for d in dims],
        "per_dimension": {d.key: _dimension_metrics(df, d, rng) for d in dims},
    }
    return metrics


def _print_dimension_summary(key: str, dm: dict) -> None:
    print("\n" + "-" * 72)
    print(f"Axis: {key}   (parsed {dm['n_parsed']}/{dm.get('n_ratings', dm['n_parsed'])})")
    print("-" * 72)
    if not dm.get("per_rating"):
        print("Nothing parsed — no metrics for this axis.")
        return

    pr = dm["per_rating"]
    print("Per-rating agreement (same person, same image):")
    print(
        f"  MAE {pr['mae']:.3f}   RMSE {pr['rmse']:.3f}   "
        f"bias {pr['mean_bias']:+.3f} (model minus human)"
    )
    print(f"  Pearson r {pr['pearson']:.3f}   Spearman rho {pr['spearman']:.3f}")
    print(
        f"  Persona-blind baselines: predict global mean -> MAE "
        f"{pr['baseline_mae_global_mean']:.3f}; predict per-image human mean -> MAE "
        f"{pr['baseline_mae_image_mean']:.3f}"
    )
    print("  (beating the per-image baseline means persona conditioning adds signal)")

    dist = dm["distribution"]
    print("\nDistribution match:")
    print(
        f"  pooled EMD {dist['pooled_emd']:.3f} score points   "
        f"pooled KS {dist['pooled_ks']:.3f}"
    )
    print(
        f"  per-image mean correlation: r {dist['image_mean_pearson']:.3f}   "
        f"rho {dist['image_mean_spearman']:.3f}"
    )
    grid = dist["score_grid"]
    fmt_hist = lambda h: "  ".join(  # noqa: E731
        f"{g:.1f}:{p * 100:4.1f}%" for g, p in zip(grid, h)
    )
    print(f"  human pooled: {fmt_hist(dist['human_hist'])}")
    print(f"  vlm pooled  : {fmt_hist(dist['vlm_hist'])}")

    print("\nPer image (EMD floor = human bootstrap resample noise):")
    header = (
        f"  {'imageName':<24} {'n':>3}  {'human m(sd)':>12}  {'vlm m(sd)':>12}"
        f"  {'MAE':>5}  {'EMD':>5}  {'floor':>5}  {'KS':>5}"
    )
    print(header)
    for row in dm["per_image"]:
        print(
            f"  {row['imageName']:<24} {row['n']:>3}"
            f"  {row['human_mean']:.2f} ({row['human_std']:.2f})"
            f"  {row['vlm_mean']:.2f} ({row['vlm_std']:.2f})"
            f"  {row['mae']:>5.2f}  {row['emd']:>5.2f}"
            f"  {row['human_resample_emd']:>5.2f}  {row['ks']:>5.2f}"
        )


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("PARA fidelity summary")
    print("=" * 72)
    print(
        f"Ratings: {metrics['n_ratings']}   "
        f"axes: {', '.join(metrics['dimensions'])}"
    )
    n_ratings = metrics["n_ratings"]
    for key in metrics["dimensions"]:
        dm = dict(metrics["per_dimension"][key])
        dm.setdefault("n_ratings", n_ratings)
        _print_dimension_summary(key, dm)


# --------------------------------------------------------------------------
# Sampling and orchestration
# --------------------------------------------------------------------------


def _van_der_corput_order(n: int) -> np.ndarray:
    """Permutation of ``range(n)`` whose every prefix is spread across ``[0, n)``.

    Ranks are ordered by the bit-reversal (van der Corput / recursive-bisection)
    of their index, so the first N entries are a low-discrepancy sample of the
    whole range and — crucially — the prefix for N is contained in the prefix for
    any larger N. Applied to score-sorted images this makes ``--n-images``
    *nested*: raising it only appends new images, never re-picks."""
    if n <= 1:
        return np.arange(n)
    bits = int(np.ceil(np.log2(n)))
    ranks = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for i in range(bits):
        rev |= ((ranks >> i) & 1) << (bits - 1 - i)
    return np.argsort(rev, kind="stable")


def choose_images(
    images_df: pd.DataFrame, n_images: int, seed: int, sampling: str, score_col: str
) -> list[str]:
    """Pick imageNames to run. 'stratified' spreads picks across the human
    mean-score range of the primary axis so even small runs cover bad-to-great
    photos; 'uniform' samples uniformly at random.

    Stratified selection is *nested*: it orders the score-sorted images by a
    van der Corput sequence and takes the first ``n_images``, so selection(N) is
    always a subset of selection(N') for N' > N. That means bumping --n-images
    reuses every image already run instead of re-picking a fresh set (seed does
    not affect the stratified set, only the uniform draw and rater subsampling)."""
    means = images_df.groupby("imageName")[score_col].mean().sort_values()
    imgs = means.index.to_numpy()
    if sampling == "uniform":
        rng = np.random.default_rng(seed)
        return list(rng.choice(imgs, size=n_images, replace=False))
    order = _van_der_corput_order(len(imgs))
    return [str(imgs[i]) for i in order[:n_images]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay PARA annotators as VLM personas on the images they rated and "
            "compare simulated vs real 1-5 aesthetic scores."
        )
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of PARA images to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated imageName values to run instead of sampling "
        "(e.g. 'iaa_pub10_.jpg,wallpaper23_.jpg').",
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
        help="Cap on how many of an image's real annotators to simulate "
        "(default: all of them, ~25 — needed for distribution matching).",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated PARA score axes to elicit and evaluate "
        f"(default: {','.join(ACTIVE_DIMENSIONS)}). "
        f"Choices: {','.join(PARA_DIMENSIONS)}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
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
        help="Ignore every rater's demographics/personality: judge each image "
        "with one generic prompt (no persona) instead of role-playing the real "
        "annotator. A model-side persona-blind baseline. Each real rater is still "
        "run (and compared against their true score), so pair with "
        "--temperature > 0 (e.g. 0.7) or every rater of an image gets an "
        "identical answer and the per-image distribution collapses to a point.",
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
        "--shard",
        default=None,
        metavar="i/N",
        help="Run only image shard i of N (0-based), so N processes — each pinned "
        "to a GPU with CUDA_VISIBLE_DEVICES — can cover disjoint images in "
        "parallel. Sharding is by image (all raters of an image stay together). "
        "Each shard's log gets a .shardIofN suffix.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON log. Defaults to data/logs/para_<timestamp>.json.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="LOG_JSON",
        help="Continue a previous (possibly crashed/partial) run: load ratings "
        "already present in this log, skip their tasks, and keep appending to "
        "the same file. Must be run with the same selection args as the "
        "original run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=60.0,
        help="Seconds between partial-progress saves to the log file while "
        "running (default: %(default)s).",
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
        "merged into combined metrics written to --output or a para_merged_* file.",
    )
    return parser.parse_args()


def _parse_shard(spec: str) -> tuple[int, int]:
    """Parse a ``i/N`` shard spec into ``(index, count)`` with validation."""
    try:
        i_str, n_str = spec.split("/")
        index, count = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(f"--shard must look like 'i/N' (e.g. 0/4), got {spec!r}")
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"--shard i/N needs 0 <= i < N and N >= 1, got {spec!r}")
    return index, count


def _shard_suffix(spec: Optional[str]) -> str:
    if not spec:
        return ""
    index, count = _parse_shard(spec)
    return f".shard{index}of{count}"


def _remap_legacy_results(results: list[dict], keys: list[str]) -> None:
    # Back-compat: older single-axis logs stored gt_score/pred_score and no
    # "dimensions" field. Remap them to the aestheticScore columns in place.
    if results and "gt_score" in results[0] and f"gt_{keys[0]}" not in results[0]:
        for r in results:
            r["gt_aestheticScore"] = r.get("gt_score")
            r["pred_aestheticScore"] = r.get("pred_score")


def analyze_log(log_paths: list[Path], output: Optional[Path] = None) -> None:
    """Recompute metrics from one or more logs.

    A single log is updated in place (unless ``output`` is given). Multiple logs
    — e.g. the per-shard outputs of a multi-GPU run — are merged: their results
    are concatenated (deduplicated by (sessionId, imageName, userId), first log
    wins) and combined metrics are written to ``output`` or a merged log file.
    """
    logs, per_log_results = [], []
    for p in log_paths:
        log, res = _read_log_and_results(p)
        logs.append(log)
        per_log_results.append(res)

    keys = logs[0].get("dimensions") or ["aestheticScore"]
    dims = resolve_dimensions(keys)

    merged: dict[tuple, dict] = {}
    for res in per_log_results:
        _remap_legacy_results(res, keys)
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
            / f"para_merged_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
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


def _task_key(record: dict) -> tuple:
    return (record["sessionId"], record["imageName"], record["userId"])


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON to ``path`` via a temp file + rename so a crash mid-write

    never leaves a truncated/corrupt log behind — the previous complete file
    stays in place until the new one is fully flushed."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _part_path(summary_path: Path, k: int) -> Path:
    """Path of the k-th (1-based) result-chunk file for a summary log."""
    return summary_path.with_name(
        f"{summary_path.stem}.part-{k:04d}{summary_path.suffix}"
    )


def _part_paths(summary_path: Path) -> list[Path]:
    """All existing result-chunk files for a summary log, in order."""
    return sorted(
        summary_path.parent.glob(f"{summary_path.stem}.part-*{summary_path.suffix}")
    )


def _n_parts(n_ratings: int, chunk_size: int) -> int:
    return (n_ratings + chunk_size - 1) // chunk_size


def _flush_chunks(
    summary_path: Path, results: list[dict], chunk_size: int, flushed_count: int
) -> int:
    """Persist result chunks that changed since ``flushed_count`` records.

    Records are partitioned into fixed-size chunks written as
    ``<stem>.part-NNNN.json``. Only the tail chunk(s) touched since the last
    flush are rewritten (each at most ``chunk_size`` records), so a checkpoint
    costs O(new records) rather than rewriting the whole run. Returns the new
    flushed count."""
    if len(results) <= flushed_count:
        return flushed_count
    first = flushed_count // chunk_size  # 0-based; may be a partially-filled chunk
    last = (len(results) - 1) // chunk_size
    for p in range(first, last + 1):
        _atomic_write_json(
            _part_path(summary_path, p + 1), results[p * chunk_size : (p + 1) * chunk_size]
        )
    return len(results)


def _read_log_and_results(summary_path: Path) -> tuple[dict, list[dict]]:
    """Load a log summary plus its ratings.

    Chunked logs (those with a ``result_parts`` key) read their ratings from the
    ``<stem>.part-NNNN.json`` files on disk; legacy single-file logs return their
    inline ``results`` list. Parts are globbed rather than trusting the manifest,
    so a crash between the last chunk write and the summary write still recovers
    every persisted rating."""
    with open(summary_path, encoding="utf-8") as f:
        log = json.load(f)
    if "result_parts" in log:
        results: list[dict] = []
        for pf in _part_paths(summary_path):
            with open(pf, encoding="utf-8") as f:
                results.extend(json.load(f))
    else:
        results = log.get("results", [])  # legacy inline format
    return log, results


def _build_summary(
    args: argparse.Namespace,
    dims: list[ScoreDimension],
    question: str,
    descriptions: dict,
    results: list[dict],
    metrics: Optional[dict],
    chunk_size: int,
    summary_path: Path,
) -> dict:
    """Build the small summary log: config, personas, metrics, and a manifest of

    the result-chunk files. The ratings themselves live in the ``.part-NNNN.json``
    files, not here, so the summary stays openable at any dataset size."""
    return {
        "dataset": "PARA",
        "image_root": str(PARA_IMAGE_DIR),
        "model_name": args.model_name,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling": args.sampling if not args.images else "explicit",
        "raters_per_image": args.raters_per_image,
        "persona_blind": args.persona_blind,
        "dimensions": [d.key for d in dims],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt_template": (
            PARA_GENERIC_SYSTEM_PROMPT if args.persona_blind else PARA_SYSTEM_PROMPT_TEMPLATE
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


def run(args: argparse.Namespace) -> None:
    if args.dimensions:
        dim_keys = [k.strip() for k in args.dimensions.split(",") if k.strip()]
    else:
        dim_keys = list(ACTIVE_DIMENSIONS)
    dims = resolve_dimensions(dim_keys)
    question = build_para_question(dims, persona_blind=args.persona_blind)
    if args.persona_blind and args.temperature == 0:
        print(
            "warning: --persona-blind with --temperature 0 gives every rater of an "
            "image the identical response (no distribution). Consider --temperature 0.7."
        )

    # sessionId/imageName/userId plus one ground-truth column per active axis.
    usecols = list(dict.fromkeys(["sessionId", "imageName", "userId", *dim_keys]))
    images_df = pd.read_csv(PARA_ANNOTATION_DIR / "PARA-Images.csv", usecols=usecols)
    users_df = pd.read_csv(PARA_ANNOTATION_DIR / "PARA-UserInfo.csv").set_index("userId")

    if args.images:
        selected = [name.strip() for name in args.images.split(",") if name.strip()]
        known = set(images_df["imageName"])
        missing = [name for name in selected if name not in known]
        if missing:
            raise ValueError(f"imageName(s) not found in PARA-Images.csv: {missing}")
    else:
        selected = choose_images(
            images_df, args.n_images, args.seed, args.sampling, dims[0].key
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
    user_index = set(users_df.index)  # O(1) membership instead of pandas Index scans
    # Group the sampled images once instead of rescanning the full 808k-row frame
    # per image (the old loop was O(n_images x len(images_df))).
    groups = {
        name: grp
        for name, grp in images_df[images_df["imageName"].isin(set(selected))].groupby(
            "imageName", sort=False
        )
    }
    tasks: list[dict] = []
    for name in selected:
        rows = groups.get(name)
        if rows is None:
            continue
        if args.raters_per_image is not None and len(rows) > args.raters_per_image:
            rows = rows.iloc[
                rng.choice(len(rows), size=args.raters_per_image, replace=False)
            ]
        for row in rows.itertuples(index=False):
            if row.userId not in user_index:
                print(f"warning: {row.userId} missing from PARA-UserInfo.csv, skipped")
                continue
            task = {
                "sessionId": row.sessionId,
                "imageName": row.imageName,
                "userId": row.userId,
            }
            for d in dims:
                task[f"gt_{d.key}"] = float(getattr(row, d.key))
            tasks.append(task)

    n_users = len({t["userId"] for t in tasks})
    print(
        f"Plan: {len(selected)} images x their real raters "
        f"= {len(tasks)} ratings from {n_users} distinct annotators."
    )

    results: list[dict] = []
    prior_users: dict = {}
    prior_chunked = False
    if args.resume:
        output_path = Path(args.resume)
        prior_log, results = _read_log_and_results(output_path)
        if prior_log.get("dimensions") != [d.key for d in dims]:
            raise ValueError(
                f"--resume log has dimensions {prior_log.get('dimensions')}, "
                f"which doesn't match --dimensions {[d.key for d in dims]}; "
                "resume with matching selection args."
            )
        if bool(prior_log.get("persona_blind", False)) != bool(args.persona_blind):
            raise ValueError(
                f"--resume log has persona_blind={prior_log.get('persona_blind', False)}, "
                f"which doesn't match --persona-blind={args.persona_blind}; "
                "persona and persona-blind ratings can't share a log."
            )
        prior_users = prior_log.get("users", {})
        prior_chunked = "result_parts" in prior_log
        # Reuse the prior chunk size so part boundaries stay consistent.
        chunk_size = int(prior_log.get("chunk_size", args.chunk_size))
        done = {_task_key(r) for r in results}
        before = len(tasks)
        tasks = [t for t in tasks if _task_key(t) not in done]
        print(
            f"Resuming from {output_path}: {before - len(tasks)} ratings already "
            f"done, {len(tasks)} remaining."
        )
    else:
        chunk_size = args.chunk_size
        # A per-shard suffix keeps parallel processes from clobbering one file.
        suffix = _shard_suffix(args.shard)
        if args.output:
            base = Path(args.output)
            output_path = base.with_name(base.stem + suffix + base.suffix)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = DEFAULT_LOG_DIR / f"para_{ts}{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptions = {
        **prior_users,
        **{
            user_id: build_para_description(users_df.loc[user_id])
            for user_id in {t["userId"] for t in tasks}
        },
    }

    if not tasks:
        print("Nothing left to do.")
        metrics = compute_metrics(results, dims, seed=args.seed)
        print_summary(metrics)
        # Result chunks already exist on disk (resume); just (re)write the summary.
        _atomic_write_json(
            output_path,
            _build_summary(args, dims, question, descriptions, results, metrics, chunk_size, output_path),
        )
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
        images = [PARA_IMAGE_DIR / t["sessionId"] / t["imageName"] for t in batch]
        if args.persona_blind:
            sys_prompts = [PARA_GENERIC_SYSTEM_PROMPT] * len(batch)
        else:
            sys_prompts = [para_system_prompt(descriptions[t["userId"]]) for t in batch]
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
    # Ratings loaded from an already-chunked resume are on disk; legacy inline
    # resumes (flushed=0) get rewritten into chunks on the first flush.
    flushed = len(results) if prior_chunked else 0
    started = time.time()
    last_checkpoint = started
    try:
        for start in range(0, total, args.batch_size):
            batch = tasks[start : start + args.batch_size]
            call_start = time.time()
            raw_responses = _run_batch(batch)
            for task, raw_response in zip(batch, raw_responses):
                pred_scores, comment = parse_para_rating(raw_response, dims)
                record = {**task, "comment": comment, "raw_response": raw_response}
                for d in dims:
                    record[f"pred_{d.key}"] = pred_scores[d.key]
                results.append(record)

            done = len(results) - n_before_run
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
                # the last checkpoint, then the (metrics-free) summary. Cheap even
                # when the run holds hundreds of thousands of ratings.
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

    flushed = _flush_chunks(output_path, results, chunk_size, flushed)
    metrics = compute_metrics(results, dims, seed=args.seed)
    print_summary(metrics)

    _atomic_write_json(
        output_path,
        _build_summary(args, dims, question, descriptions, results, metrics, chunk_size, output_path),
    )
    print(f"\nWrote log to {output_path}")


def main() -> None:
    args = _parse_args()
    if args.analyze_only:
        paths = [Path(p.strip()) for p in args.analyze_only.split(",") if p.strip()]
        analyze_log(paths, output=Path(args.output) if args.output else None)
        return
    run(args)


if __name__ == "__main__":
    main()
