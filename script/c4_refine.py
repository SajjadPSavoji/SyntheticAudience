"""C4 driver — auto-refinement editing loop (society vs blind vs static critic).

Selects low-to-mid-rated source images from EVA and/or PARA, builds a fixed
PARA persona panel, then runs the 10-step anchored-re-edit / accept-if-better
loop (``src/editor/loop.py``) for each (image, condition). Edited images land
under ``data/c4_edits/``; per-step logs are written as sharded/chunked JSON
under ``data/results/c4_<condition>/`` (same layout as the persona runs, so the
analysis tooling reads them the same way).

Heavy imports (torch/diffusers) are deferred until a run actually needs a GPU.

Example (A100 Colab):
    python script/c4_refine.py --dataset both --n-images 150 \
        --conditions static,blind,society,reward_only \
        --editor flux --steps 10 --candidates 3
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "script"))
import para_pipeline as para  # torch-free: builders, parser, chunked-IO, van der Corput

# Verified local data layout (NOT the pipeline path constants, which point at
# nonexistent data/PARA and data/eva-dataset dirs).
DATA = REPO_ROOT / "data"
EVA_IMAGES = DATA / "eva" / "images"
EVA_VOTES = DATA / "eva" / "data" / "votes_filtered.csv"
PARA_IMAGES = DATA / "para" / "imgs"
PARA_IMAGES_CSV = DATA / "para" / "annotation" / "PARA-Images.csv"
PARA_USERINFO = DATA / "para" / "annotation" / "PARA-UserInfo.csv"

EDITS_DIR = DATA / "c4_edits"
RESULTS_DIR = DATA / "results"


# --------------------------------------------------------------------------
# Source-image selection (low-to-mid human-rated band, nested/reproducible)
# --------------------------------------------------------------------------

def _banded_nested(means: pd.Series, band: tuple[float, float], n: int) -> list[str]:
    """Take the [band_lo, band_hi] percentile slice of per-image means and order
    it with the van der Corput sequence so ``n`` is nested (raising it appends)."""
    means = means.sort_values()
    lo_q, hi_q = np.quantile(means.to_numpy(), list(band))
    keep = means[(means >= lo_q) & (means <= hi_q)]
    order = para._van_der_corput_order(len(keep))
    idx = keep.index.to_numpy()
    return [idx[i] for i in order[: min(n, len(idx))]]  # native key type (int for EVA, str for PARA)


def select_eva(n: int, band: tuple[float, float]) -> list[dict]:
    df = pd.read_csv(EVA_VOTES, sep="=", usecols=["image_id", "score"])
    means = df.groupby("image_id")["score"].mean()
    chosen = _banded_nested(means, band, n)
    out = []
    for img in chosen:
        p = EVA_IMAGES / f"{img}.jpg"
        if p.exists():
            out.append({"image_id": f"eva__{img}", "path": str(p),
                        "dataset": "eva", "human_mean": float(means[img])})
    return out


def select_para(n: int, band: tuple[float, float]) -> list[dict]:
    df = pd.read_csv(PARA_IMAGES_CSV, usecols=["sessionId", "imageName", "aestheticScore"])
    means = df.groupby("imageName")["aestheticScore"].mean()
    session_of = df.drop_duplicates("imageName").set_index("imageName")["sessionId"]
    chosen = _banded_nested(means, band, n)
    out = []
    for img in chosen:
        p = PARA_IMAGES / str(session_of[img]) / img
        if p.exists():
            out.append({"image_id": f"para__{img}", "path": str(p),
                        "dataset": "para", "human_mean": float(means[img])})
    return out


def select_source_images(dataset: str, n: int, band: tuple[float, float]) -> list[dict]:
    imgs: list[dict] = []
    if dataset in ("eva", "both"):
        imgs += select_eva(n, band)
    if dataset in ("para", "both"):
        imgs += select_para(n, band)
    return imgs


def build_panel(panel_size: int, seed: int) -> list[str]:
    """Sample a fixed panel of PARA personas (rich Big-Five + art/photo exp)."""
    users = pd.read_csv(PARA_USERINFO)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(users), size=min(panel_size, len(users)), replace=False)
    return [para.build_para_description(users.iloc[i]) for i in sorted(take)]


# --------------------------------------------------------------------------
# Running one condition
# --------------------------------------------------------------------------

def _shard(items: list, spec: str | None) -> list:
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    return items[i::n]


def _done_image_ids(summary_path: Path) -> set:
    if not summary_path.exists():
        return set()
    try:
        _, rows = para._read_log_and_results(summary_path)
        return {r["image_id"] for r in rows}
    except Exception:
        return set()


def run_condition(condition, images, *, args, panel, backend, editor, objective, drift):
    from editor import EditCache, build_critic, distill_instruction, run_refinement
    from editor.loop import records_to_dicts

    run_name = f"c4_{condition}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = "" if not args.shard else f".shard{args.shard.replace('/', 'of')}"
    summary_path = out_dir / f"{run_name}{shard_tag}.json"

    done = _done_image_ids(summary_path) if args.resume else set()
    rows: list[dict] = []
    if done:  # keep previously flushed rows so chunk offsets stay consistent
        _, rows = para._read_log_and_results(summary_path)
    flushed = len(rows)

    critic = build_critic(condition, backend=backend, panel=panel)
    distill = lambda img, acc, comps: distill_instruction(  # noqa: E731
        backend, img, acc, comps, max_words=args.max_instruction_words)
    cache = EditCache(str(EDITS_DIR / condition / "_cache.json"))

    todo = [im for im in images if im["image_id"] not in done]
    print(f"[{condition}] {len(todo)} images to run ({len(done)} already done)")
    for n_done, im in enumerate(todo, 1):
        records = run_refinement(
            im["image_id"], im["path"],
            condition=condition, editor=editor, objective=objective, drift=drift,
            critic=critic, distill=distill, save_dir=str(EDITS_DIR),
            R=args.steps, K=args.candidates, drift_cap=args.drift_cap,
            seed=args.seed, cache=cache,
        )
        step_rows = records_to_dicts(im["image_id"], condition, records)
        for row in step_rows:
            row["dataset"] = im["dataset"]
            row["human_mean"] = im["human_mean"]
        rows.extend(step_rows)

        if n_done % args.checkpoint_interval == 0 or n_done == len(todo):
            flushed = para._flush_chunks(summary_path, rows, para.DEFAULT_CHUNK_SIZE, flushed)
            _write_summary(summary_path, condition, args, panel, rows)
            print(f"  [{condition}] checkpoint: {n_done}/{len(todo)} images, {len(rows)} rows")


def _write_summary(summary_path: Path, condition, args, panel, rows):
    summary = {
        "run": f"c4_{condition}",
        "condition": condition,
        "is_oracle": condition == "reward_only",
        "editor": args.editor,
        "editor_model": args.model_name if args.editor == "flux" else args.editor,
        "critic_model": args.critic_model,
        "objective_model": "improved-aesthetic-predictor (CLIP ViT-L/14)",
        "drift_backbone": "facebook/dinov2-base",
        "drift_cap": args.drift_cap,
        "steps": args.steps,
        "candidates": args.candidates,
        "panel_size": len(panel),
        "panel": panel,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_size": para.DEFAULT_CHUNK_SIZE,
        "n_ratings": len(rows),
        "result_parts": [
            para._part_path(summary_path, k).name
            for k in range(1, para._n_parts(len(rows), para.DEFAULT_CHUNK_SIZE) + 1)
        ],
    }
    para._atomic_write_json(summary_path, summary)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C4 auto-refinement editing loop.")
    p.add_argument("--dataset", choices=["eva", "para", "both"], default="both")
    p.add_argument("--n-images", type=int, default=150,
                   help="images per dataset from the low-to-mid band (default 150).")
    p.add_argument("--low-mid-band", type=float, nargs=2, default=(0.15, 0.60),
                   metavar=("LO", "HI"), help="percentile band of human mean score.")
    p.add_argument("--conditions", default="static,blind,society,reward_only")
    p.add_argument("--editor", choices=["flux", "instructpix2pix"], default="flux")
    p.add_argument("--model-name", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--critic-model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--drift-cap", type=float, default=0.85)
    p.add_argument("--panel-size", type=int, default=10)
    p.add_argument("--panel-seed", type=int, default=0)
    p.add_argument("--max-instruction-words", type=int, default=15)
    p.add_argument("--cpu-offload", action="store_true",
                   help="stream FLUX weights (use on <40GB GPUs; not needed on A100).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", default=None, metavar="i/N",
                   help="run image sub-shard i of N (round-robin).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    images = select_source_images(args.dataset, args.n_images, tuple(args.low_mid_band))
    images = _shard(images, args.shard)
    if not images:
        print("No source images selected — check data/ layout.")
        return
    panel = build_panel(args.panel_size, args.panel_seed)
    print(f"Selected {len(images)} source images; panel of {len(panel)} personas; "
          f"conditions={conditions}")

    # Deferred heavy construction (only now do we need a GPU).
    needs_vlm = any(c in ("blind", "society") for c in conditions)
    backend = None
    if needs_vlm:
        print(f"Loading critic backend '{args.critic_model}'...")
        from persona import QwenVLBackend
        backend = QwenVLBackend(model_name=args.critic_model)

    from editor import AestheticObjective, DriftMetric, build_editor
    print(f"Loading editor '{args.editor}'...")
    editor_kwargs = {}
    if args.editor == "flux":
        editor_kwargs["model_name"] = args.model_name
        editor_kwargs["cpu_offload"] = args.cpu_offload
    editor = build_editor(args.editor, **editor_kwargs)
    print("Loading aesthetic objective + drift metric...")
    objective = AestheticObjective()
    drift = DriftMetric()

    for condition in conditions:
        run_condition(condition, images, args=args, panel=panel, backend=backend,
                      editor=editor, objective=objective, drift=drift)
    print("Done.")


if __name__ == "__main__":
    main()
