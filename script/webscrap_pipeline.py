"""Webscrap pipeline: judge an *unlabeled* image folder with the PARA model.

data/webscrap/ is a scraped image folder with no annotations — no raters, no
ground-truth scores. So this script is deliberately the persona-blind arm of
script/para_pipeline.py, run standalone: every image is judged with the generic,
rater-agnostic PARA_GENERIC_SYSTEM_PROMPT rather than in the voice of any
annotator, ``--samples-per-image`` times at ``--temperature > 0``. The repeated
samples are what give each image *multiple scores*; their spread is decoding
noise, not taste diversity, since nothing conditions the model on who is judging.

What carries over from para_pipeline.py, unchanged: the ScoreDimension grids and
their prompt glosses, the generic system prompt, the question builder, and the
response parser. Metrics are imported where they still have a referent and
recomputed where they don't:

  * Anything ground-truth-relative in PARA — MAE / RMSE / bias against the real
    rater, EMD / KS against the human distribution, the bootstrap human-resample
    noise floor, the predict-global-mean and predict-image-mean baselines — has
    no referent here and is *not* reported. There is nothing to be right about.
  * What survives is the distribution machinery, with the corpus standing in for
    the humans: per-image mean/sd/histogram per axis, the pooled histogram over
    all images, and each image's EMD/KS *against the pooled corpus distribution*
    (how unusual this image's scores are for this model), scaled by a bootstrap
    floor computed the same way human_resample_emd computes PARA's.
  * Two things PARA cannot measure but an unlabeled corpus can: within-image
    sampling spread vs between-image spread of the means (a discrimination
    ratio — if the model can't separate images by more than it disagrees with
    itself, the scores carry little signal), and cross-axis correlation of the
    per-image means (do the axes collapse onto one latent "is it good").

Because the spread across samples is the *only* dispersion this design has, the
decoding parameters are load-bearing. Qwen2-VL-7B-Instruct's shipped
generation_config.json sets top_k=1 and top_p=0.001; transformers merges those
into every generate() call, and top_k=1 is argmax, so passing do_sample=True and
a temperature is not by itself enough to get sampling — the run silently returns
byte-identical responses. This script therefore always passes --top-k/--top-p
explicitly (defaults 0 / 1.0 = pure temperature sampling).

Unlike the dataset-fidelity pipelines there is no sharding and no chunked log:
one run writes one JSON file, keyed by image name.

Run from the repo root (persona conda env):

    python script/webscrap_pipeline.py --samples-per-image 8 --temperature 0.7
    python script/webscrap_pipeline.py --n-images 5 --dimensions aestheticScore
    python script/webscrap_pipeline.py --analyze-only data/logs/webscrap_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make ``src`` (persona package) and ``script`` (para_pipeline) importable when
# this file is run from anywhere, not just via ``python script/webscrap_pipeline.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "script"))

from para_pipeline import (  # noqa: E402  (needs sys.path tweak above)
    PARA_DIMENSIONS,
    PARA_GENERIC_SYSTEM_PROMPT,
    ScoreDimension,
    _atomic_write_json,
    _hist_on_grid,
    _pearson,
    _spearman,
    build_para_question,
    emd,
    generate_with_retry,
    ks_stat,
    parse_para_rating as parse_rating,
    resolve_dimensions,
)

WEBSCRAP_IMAGE_DIR = REPO_ROOT / "data" / "webscrap"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# All PARA axes by default: the images are unlabeled, so there is no reason to
# narrow to the one axis a ground-truth column happens to exist for.
ACTIVE_DIMENSIONS = list(PARA_DIMENSIONS)


def list_images(image_dir: Path) -> list[str]:
    """Image filenames in ``image_dir``, sorted for a reproducible task order.

    Sorted by name rather than by st_mtime/os.listdir order so that --n-images
    and --resume select the same prefix across runs and machines.
    """
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# --------------------------------------------------------------------------
# Metrics: no labels, so everything is relative to the corpus or to itself
# --------------------------------------------------------------------------


def bootstrap_emd_floor(scores, rng, grid: np.ndarray, n_boot: int = 200) -> float:
    """Expected EMD between a sample and a bootstrap resample of itself.

    The analogue of para_pipeline.human_resample_emd, but with the model's own
    samples in place of the human raters: it is the EMD this image would show
    against itself purely from having drawn only ``len(scores)`` samples. An
    ``emd_vs_pooled`` at or below this floor means the image is statistically
    indistinguishable from the corpus as a whole.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return float("nan")
    return float(
        np.mean([emd(rng.choice(s, size=s.size, replace=True), s, grid) for _ in range(n_boot)])
    )


def _dimension_metrics(df: pd.DataFrame, dim: ScoreDimension, rng) -> dict:
    """Corpus-relative metrics for one axis over its ``pred_<key>`` column."""
    pred_col = f"pred_{dim.key}"
    grid = dim.grid
    out: dict = {"n_parsed": 0}
    if len(df) == 0 or pred_col not in df.columns:
        return out
    parsed = df[df[pred_col].notna()]
    if len(parsed) == 0:
        return out
    out["n_parsed"] = int(len(parsed))
    out["parse_rate"] = float(len(parsed) / len(df))

    pooled = parsed[pred_col].to_numpy(dtype=float)

    per_image = []
    for name, grp in parsed.groupby("imageName", sort=False):
        s = grp[pred_col].to_numpy(dtype=float)
        per_image.append(
            {
                "imageName": name,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "emd_vs_pooled": emd(s, pooled, grid),
                "ks_vs_pooled": ks_stat(s, pooled, grid),
                "bootstrap_emd_floor": bootstrap_emd_floor(s, rng, grid),
            }
        )
    out["per_image"] = per_image

    pi = pd.DataFrame(per_image)
    # Within = how much the model disagrees with itself on one image (decoding
    # noise). Between = how much the per-image means move apart. Their ratio is
    # the axis's usable resolution: <= 1 means the axis barely separates images.
    within = float(np.sqrt((pi["std"] ** 2).mean()))
    between = float(pi["mean"].std(ddof=1)) if len(pi) > 1 else 0.0
    out["distribution"] = {
        "pooled_mean": float(pooled.mean()),
        "pooled_std": float(pooled.std(ddof=1)) if len(pooled) > 1 else 0.0,
        "score_grid": grid.tolist(),
        "pooled_hist": _hist_on_grid(pooled, grid).tolist(),
        "grid_coverage": float((_hist_on_grid(pooled, grid) > 0).mean()),
        "within_image_std": within,
        "between_image_std": between,
        "discrimination_ratio": float(between / within) if within else float("nan"),
    }
    return out


def _cross_dimension(df: pd.DataFrame, dims: list[ScoreDimension]) -> dict:
    """Pearson/Spearman between per-image means of each pair of axes.

    Near-1 correlations everywhere mean the axes are not measuring distinct
    things — the model is answering one question and relabelling the answer.
    """
    if len(dims) < 2:
        return {}
    means = df.groupby("imageName")[[f"pred_{d.key}" for d in dims]].mean()
    out = {}
    for i, a in enumerate(dims):
        for b in dims[i + 1 :]:
            pair = means[[f"pred_{a.key}", f"pred_{b.key}"]].dropna()
            if len(pair) < 2:
                continue
            x = pair[f"pred_{a.key}"].to_numpy()
            y = pair[f"pred_{b.key}"].to_numpy()
            out[f"{a.key}~{b.key}"] = {
                "n_images": int(len(pair)),
                "pearson": _pearson(x, y),
                "spearman": _spearman(x, y),
            }
    return out


def compute_metrics(records: list[dict], dims: list[ScoreDimension], seed: int = 0) -> dict:
    """Corpus-relative metrics per axis over per-sample records."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records)
    return {
        "n_samples": int(len(df)),
        "n_images": int(df["imageName"].nunique()) if len(df) else 0,
        "dimensions": [d.key for d in dims],
        "per_dimension": {d.key: _dimension_metrics(df, d, rng) for d in dims},
        "cross_dimension": _cross_dimension(df, dims) if len(df) else {},
    }


def _print_dimension_summary(key: str, dm: dict, n_samples: int, top_k: int = 5) -> None:
    print("\n" + "-" * 72)
    print(f"Axis: {key}   (parsed {dm['n_parsed']}/{n_samples})")
    print("-" * 72)
    if not dm.get("per_image"):
        print("Nothing parsed — no metrics for this axis.")
        return

    dist = dm["distribution"]
    print(
        f"Pooled: mean {dist['pooled_mean']:.3f}  sd {dist['pooled_std']:.3f}  "
        f"grid coverage {dist['grid_coverage'] * 100:.0f}% of bins used"
    )
    ratio = dist["discrimination_ratio"]
    ratio_str = f"{ratio:.2f}x" if np.isfinite(ratio) else "undefined (no within-image spread)"
    print(
        f"  within-image sd {dist['within_image_std']:.3f}   "
        f"between-image sd {dist['between_image_std']:.3f}   "
        f"discrimination {ratio_str}"
    )
    if not np.isfinite(ratio):
        print("  (every sample of each image was identical — decoding is greedy; "
              "check --temperature / --top-k)")
    else:
        print("  (discrimination <= 1 means the model separates images no better "
              "than it disagrees with itself)")
    grid, hist = dist["score_grid"], dist["pooled_hist"]
    print("  pooled: " + "  ".join(f"{g:g}:{p * 100:4.1f}%" for g, p in zip(grid, hist)))

    pi = sorted(dm["per_image"], key=lambda r: r["mean"], reverse=True)
    # Only elide the middle when there is a middle to elide; otherwise a small
    # corpus would print every image twice (once as "top", once as "bottom").
    if len(pi) > 2 * top_k:
        rows = pi[:top_k] + [None] + pi[-top_k:]
        print(f"\nTop {top_k} / bottom {top_k} images by mean {key}:")
    else:
        rows = pi
        print(f"\nAll {len(pi)} images by mean {key}:")
    print(
        f"  {'imageName':<44} {'n':>3}  {'mean':>5}  {'sd':>5}"
        f"  {'EMDvsPool':>9}  {'floor':>5}  {'KS':>5}"
    )
    for row in rows:
        if row is None:
            print(f"  {'...':<44} ({len(pi) - 2 * top_k} more)")
            continue
        name = row["imageName"]
        name = name if len(name) <= 44 else name[:41] + "..."
        print(
            f"  {name:<44} {row['n']:>3}  {row['mean']:>5.2f}  {row['std']:>5.2f}"
            f"  {row['emd_vs_pooled']:>9.3f}  {row['bootstrap_emd_floor']:>5.3f}"
            f"  {row['ks_vs_pooled']:>5.2f}"
        )


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("Webscrap persona-blind summary (no ground truth — corpus-relative only)")
    print("=" * 72)
    print(
        f"Samples: {metrics['n_samples']} over {metrics['n_images']} images   "
        f"axes: {', '.join(metrics['dimensions'])}"
    )
    for key in metrics["dimensions"]:
        _print_dimension_summary(key, metrics["per_dimension"][key], metrics["n_samples"])

    cross = metrics.get("cross_dimension")
    if cross:
        print("\n" + "-" * 72)
        print("Cross-axis correlation of per-image means")
        print("-" * 72)
        for pair, st in sorted(cross.items(), key=lambda kv: -abs(kv[1]["pearson"])):
            print(f"  {pair:<44} r {st['pearson']:+.3f}  rho {st['spearman']:+.3f}")
        print("  (uniformly high r means the axes collapse to one latent judgement)")


# --------------------------------------------------------------------------
# Log shape: one JSON file, keyed by image name
# --------------------------------------------------------------------------


def _build_log(
    args: argparse.Namespace,
    dims: list[ScoreDimension],
    question: str,
    records: list[dict],
    metrics: Optional[dict],
) -> dict:
    """Assemble the single output document.

    ``images`` is the part the user asked for: image name -> the several scores
    that image received on each axis, plus the raw samples behind them. No
    derived per-image stats are stored — mean/sd/etc. are a one-liner over
    ``scores`` and live in ``metrics.per_dimension.<axis>.per_image`` anyway.
    """
    images: dict[str, dict] = {}
    for r in records:
        img = images.setdefault(
            r["imageName"],
            {"n_samples": 0, "scores": {d.key: [] for d in dims}, "samples": []},
        )
        img["n_samples"] += 1
        for d in dims:
            img["scores"][d.key].append(r[f"pred_{d.key}"])
        img["samples"].append(
            {
                "sample": r["sample"],
                "comment": r["comment"],
                "raw_response": r["raw_response"],
            }
        )

    return {
        "dataset": "webscrap",
        "image_root": str(WEBSCRAP_IMAGE_DIR),
        "labeled": False,
        "persona_blind": True,  # the only mode: there are no raters to role-play
        "model_name": args.model_name,
        "backend": args.backend,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "samples_per_image": args.samples_per_image,
        "seed": args.seed,
        "dimensions": [d.key for d in dims],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": PARA_GENERIC_SYSTEM_PROMPT,
        "question": question,
        "n_samples": len(records),
        "n_images": len(images),
        "images": images,
        "metrics": metrics,
    }


def _records_from_log(log: dict, dims: list[ScoreDimension]) -> list[dict]:
    """Flatten the ``images`` map back into per-sample records for metrics."""
    records = []
    for name, img in log.get("images", {}).items():
        for i, sample in enumerate(img["samples"]):
            rec = {"imageName": name, "sample": sample["sample"]}
            for d in dims:
                rec[f"pred_{d.key}"] = img["scores"][d.key][i]
            records.append(rec)
    return records


def analyze_log(path: Path, output: Optional[Path] = None) -> None:
    """Recompute metrics from an existing log without touching the model."""
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    dims = resolve_dimensions(log.get("dimensions") or ACTIVE_DIMENSIONS)
    records = _records_from_log(log, dims)
    metrics = compute_metrics(records, dims, seed=int(log.get("seed", 0)))
    log["metrics"] = metrics
    target = output or path
    _atomic_write_json(target, log)
    print(f"Recomputed metrics from {path} -> {target} ({metrics['n_samples']} samples).")
    print_summary(metrics)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge an unlabeled image folder with the persona-blind PARA prompts, "
            "sampling each image several times to get a score distribution."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=WEBSCRAP_IMAGE_DIR,
        help="Folder of images to judge (default: %(default)s).",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Judge only the first N images (name-sorted). Default: all of them.",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated image filenames to run instead of scanning the folder.",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=8,
        help="How many independent judgements to draw per image; these are the "
        "'multiple scores' each image ends up with (default: %(default)s).",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated score axes to elicit "
        f"(default: {','.join(ACTIVE_DIMENSIONS)}). Choices: {','.join(PARA_DIMENSIONS)}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default: 0).")
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
        default=0.7,
        help="Sampling temperature. Must be > 0 or every sample of an image is "
        "byte-identical and the per-image distribution collapses to a point "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k truncation; 0 disables it. Must not be left at Qwen2-VL's "
        "shipped default of 1, which is argmax and makes --temperature inert "
        "(default: %(default)s = pure temperature sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus truncation; 1.0 disables it. Qwen2-VL ships 0.001 "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation budget per sample (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many samples to run through the model in one padded generate() "
        "call (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON log. Defaults to data/logs/webscrap_<timestamp>.json.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=60.0,
        help="Seconds between partial-progress saves while running (default: %(default)s).",
    )
    parser.add_argument(
        "--analyze-only",
        default=None,
        metavar="LOG_JSON",
        help="Skip the model: recompute metrics from an existing log and print the summary.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    dim_keys = (
        [k.strip() for k in args.dimensions.split(",") if k.strip()]
        if args.dimensions
        else list(ACTIVE_DIMENSIONS)
    )
    dims = resolve_dimensions(dim_keys)
    question = build_para_question(dims, persona_blind=True)

    if args.temperature <= 0:
        raise SystemExit(
            "--temperature must be > 0: with greedy decoding every sample of an "
            "image is identical, so the per-image distribution is a single point. "
            "Try --temperature 0.7."
        )
    if args.samples_per_image < 2:
        print(
            "warning: --samples-per-image < 2 gives no within-image spread, so the "
            "distribution metrics (sd, EMD floor, discrimination) are degenerate."
        )

    if args.images:
        selected = [n.strip() for n in args.images.split(",") if n.strip()]
        missing = [n for n in selected if not (args.image_dir / n).is_file()]
        if missing:
            raise SystemExit(f"image(s) not found in {args.image_dir}: {missing}")
    else:
        selected = list_images(args.image_dir)
        if not selected:
            raise SystemExit(f"no images found in {args.image_dir}")
        if args.n_images is not None:
            selected = selected[: args.n_images]

    # Samples of one image sit adjacent in the task list so a batch usually
    # covers a single image — the encoder then sees one repeated image per call.
    tasks = [
        {"imageName": name, "sample": k}
        for name in selected
        for k in range(args.samples_per_image)
    ]
    print(
        f"Plan: {len(selected)} images x {args.samples_per_image} samples "
        f"= {len(tasks)} judgements across {len(dims)} axes."
    )

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_LOG_DIR / f"webscrap_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.backend} backend '{args.model_name}'...")
    from persona import LlavaBackend, QwenVLBackend  # deferred: pulls in torch

    backend_cls = QwenVLBackend if args.backend == "qwen" else LlavaBackend
    backend = backend_cls(model_name=args.model_name)

    # top_k / top_p must be set explicitly: Qwen2-VL-7B-Instruct ships a
    # generation_config.json with top_k=1, top_p=0.001, which HF merges into every
    # generate() call. top_k=1 is argmax, so passing temperature alone leaves
    # decoding greedy and every sample of an image comes back byte-identical.
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    def _run_batch(batch: list[dict]) -> list[str]:
        images = [args.image_dir / t["imageName"] for t in batch]
        sys_prompts = [PARA_GENERIC_SYSTEM_PROMPT] * len(batch)
        return generate_with_retry(
            backend, sys_prompts, images, [question] * len(batch), gen_kwargs
        )

    records: list[dict] = []
    total = len(tasks)
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
                records.append(record)

            done = len(records)
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
                _atomic_write_json(output_path, _build_log(args, dims, question, records, None))
                last_checkpoint = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(records)}/{total} judgements — writing partial log.")

    elapsed = time.time() - started
    print(f"\nGenerated {len(records)} judgements in {elapsed / 60:.1f} min.")

    metrics = compute_metrics(records, dims, seed=args.seed)
    print_summary(metrics)
    _atomic_write_json(output_path, _build_log(args, dims, question, records, metrics))
    print(f"\nWrote log to {output_path}")


def main() -> None:
    args = _parse_args()
    if args.analyze_only:
        analyze_log(Path(args.analyze_only), output=Path(args.output) if args.output else None)
        return
    run(args)


if __name__ == "__main__":
    main()
