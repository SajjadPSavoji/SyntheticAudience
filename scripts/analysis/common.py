"""Shared loading + normalization for the teammate's result logs (data/results/*).

Every run is a sharded export: per-shard summary files ``<run>.shardNof4.json``
(config, dims, per-run ``users`` persona cards, and a precomputed ``metrics`` block)
plus the actual per-(image,user) records in ``<run>.shardNof4.part-KKKK.json`` chunks.

This module loads a run into one tidy DataFrame with an added ``*_norm`` column per
dimension on a common [0,1] scale, so every downstream analysis is scale-agnostic.

No GPU / no inference — pure re-analysis of the cached logs.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO, "data", "results")
OUT_DIR = os.path.join(REPO, "results")

# Native rating scale (lo, hi) per (dataset, dimension), read from the pipeline
# ScoreDimension definitions. Normalization to [0,1] is (x - lo) / (hi - lo).
SCALES: dict[str, dict[str, tuple[float, float]]] = {
    "PARA": {k: (1.0, 5.0) for k in [
        "aestheticScore", "qualityScore", "compositionScore", "colorScore",
        "dofScore", "contentScore", "lightScore", "contentPreference",
        "willingnessToShare"]},
    "EVA": {"score": (0.0, 10.0), "visual": (1.0, 4.0), "composition": (1.0, 4.0),
            "quality": (1.0, 4.0), "semantic": (1.0, 4.0), "difficulty": (1.0, 4.0)},
    "LAPIS": {"rating": (0.0, 100.0)},
}

# The headline dimension per dataset (the one the claims center on).
PRIMARY_DIM = {"PARA": "aestheticScore", "EVA": "score", "LAPIS": "rating"}

RUNS = ["para_full", "para_blind", "eva_full", "eva_blind", "lapis_full", "lapis_blind"]


@dataclass
class Run:
    name: str                    # e.g. "para_full"
    dataset: str                 # PARA / EVA / LAPIS
    mode: str                    # "full" (persona) or "blind"
    config: dict                 # merged run config (from a summary shard)
    dims: list[str]              # active dimensions
    df: pd.DataFrame             # one row per (image, user) with gt_/pred_/*_norm
    users: dict = field(default_factory=dict)   # userId -> persona card text
    baked_metrics: list[dict] = field(default_factory=list)  # per-shard metrics blocks


def _run_dir(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def _summary_shards(name: str) -> list[str]:
    d = _run_dir(name)
    return sorted(f for f in glob.glob(os.path.join(d, f"{name}.shard*of*.json"))
                  if ".part-" not in os.path.basename(f))


def _part_files(name: str) -> list[str]:
    d = _run_dir(name)
    return sorted(glob.glob(os.path.join(d, f"{name}.shard*.part-*.json")))


def load_run(name: str) -> Run:
    """Load one run folder into a normalized Run object."""
    summaries = _summary_shards(name)
    if not summaries:
        raise FileNotFoundError(f"no summary shards for run {name} under {_run_dir(name)}")

    config = None
    dims: list[str] = []
    users: dict = {}
    baked: list[dict] = []
    for s in summaries:
        d = json.load(open(s, encoding="utf-8"))
        if config is None:
            config = {k: v for k, v in d.items()
                      if k not in ("users", "metrics", "result_parts")}
            dims = d.get("dimensions") or ["aestheticScore"]
        users.update(d.get("users", {}))
        if "metrics" in d:
            baked.append(d["metrics"])

    dataset = config.get("dataset", "").upper()
    mode = "blind" if config.get("persona_blind") else "full"

    records: list[dict] = []
    for p in _part_files(name):
        records.extend(json.load(open(p, encoding="utf-8")))
    df = pd.DataFrame.from_records(records)

    # Normalize every present dimension to [0,1] as <dim>_gt_norm / <dim>_pred_norm.
    scale = SCALES[dataset]
    for dim, (lo, hi) in scale.items():
        gt_c, pred_c = f"gt_{dim}", f"pred_{dim}"
        if gt_c in df.columns:
            df[f"{dim}_gt_norm"] = (pd.to_numeric(df[gt_c], errors="coerce") - lo) / (hi - lo)
        if pred_c in df.columns:
            df[f"{dim}_pred_norm"] = (pd.to_numeric(df[pred_c], errors="coerce") - lo) / (hi - lo)

    return Run(name=name, dataset=dataset, mode=mode, config=config,
               dims=dims, df=df, users=users, baked_metrics=baked)


def load_all() -> dict[str, Run]:
    return {name: load_run(name) for name in RUNS}


def ensure_out() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def write_json(obj, filename: str) -> str:
    ensure_out()
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
