"""Part A audit — validate the result logs before trusting any number.

Runs the checks from docs/analysis_protocol.md Part A against data/results/*:
  A1 reproduce the baked-in per_rating metrics from raw (pooled across shards)
  A2 parse-failure rate + concentration
  A4 coverage: full/blind pairing, raters-per-image distribution
  A7 degenerate-prediction fraction (vlm_std==0 proxy: identical preds per image)
  A9 manifest completeness: row counts vs baked n_ratings, duplicate keys
  A8 baseline labelling note (image-mean == group oracle, not a fair floor)

Writes results/audit.json and prints a human-readable summary. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (PRIMARY_DIM, RUNS, Run, load_run, write_json)

KEY = ["imageName", "userId"]


def _pooled_baked(run: Run, dim: str) -> dict:
    """n-weighted pool of the per-shard baked per_rating metrics for one dim."""
    tot = 0.0
    acc = {k: 0.0 for k in ("mae", "spearman", "pearson", "mean_bias",
                            "baseline_mae_global_mean", "baseline_mae_image_mean")}
    for m in run.baked_metrics:
        pd_ = m.get("per_dimension", {}).get(dim)
        if not pd_:
            continue
        n = pd_["n_parsed"]
        tot += n
        for k in acc:
            acc[k] += pd_["per_rating"][k] * n
    if tot == 0:
        return {}
    return {k: v / tot for k, v in acc.items()} | {"n_parsed": int(tot)}


def _recompute(df: pd.DataFrame, dim: str) -> dict:
    """Recompute per_rating metrics from raw records on the NATIVE scale."""
    gt, pred = df[f"gt_{dim}"], df[f"pred_{dim}"]
    m = pd.to_numeric(pred, errors="coerce").notna() & pd.to_numeric(gt, errors="coerce").notna()
    gt, pred = pd.to_numeric(gt[m]), pd.to_numeric(pred[m])
    n = len(gt)
    err = (pred - gt).abs()
    global_mean = gt.mean()
    image_mean = df.loc[m].groupby("imageName")[f"gt_{dim}"].transform("mean")
    out = {
        "n_parsed": int(n),
        "mae": float(err.mean()),
        "mean_bias": float((pred - gt).mean()),
        "spearman": float(stats.spearmanr(pred, gt).statistic),
        "pearson": float(stats.pearsonr(pred, gt).statistic),
        "baseline_mae_global_mean": float((gt - global_mean).abs().mean()),
        "baseline_mae_image_mean": float((gt - image_mean).abs().mean()),
    }
    return out


def audit_run(run: Run) -> dict:
    df = run.df
    dim = PRIMARY_DIM[run.dataset]
    pred_col, gt_col = f"pred_{dim}", f"gt_{dim}"

    # A9 manifest / duplicates
    n_rows = len(df)
    dup = int(df.duplicated(subset=KEY).sum())
    baked_n = sum(m.get("n_ratings", 0) for m in run.baked_metrics)

    # Break duplicate (image,user) pairs into exact-identical rows (export artifact)
    # vs genuine repeated measures (same rater, same image, DIFFERENT gt score).
    dup_detail = {"dup_pairs": 0, "pairs_same_gt": 0, "pairs_differing_gt": 0,
                  "exact_identical_rows": 0}
    if dup:
        dmask = df.duplicated(subset=KEY, keep=False)
        d = df[dmask]
        by = d.groupby(KEY)[gt_col].nunique()
        dup_detail["dup_pairs"] = int(len(by))
        dup_detail["pairs_same_gt"] = int((by == 1).sum())
        dup_detail["pairs_differing_gt"] = int((by > 1).sum())
        dup_detail["exact_identical_rows"] = int(
            df.duplicated(subset=[c for c in df.columns if not c.endswith("_norm")]).sum())

    # A2 parse failures (primary dim as proxy; small VLMs fail JSON occasionally)
    pred_num = pd.to_numeric(df[pred_col], errors="coerce")
    n_fail = int(pred_num.isna().sum())
    fail_frac = n_fail / n_rows if n_rows else 0.0
    # concentration: is failure spread evenly or clustered on few images?
    fail_by_img = df.assign(_f=pred_num.isna()).groupby("imageName")["_f"].mean()
    fail_img_share = float((fail_by_img > 0).mean())

    # A7 degenerate predictions: images where all personas give the identical pred
    g = df.dropna(subset=[pred_col]).groupby("imageName")[pred_col]
    per_img_nunique = g.nunique()
    degen_frac = float((per_img_nunique <= 1).mean())
    mean_raters = float(df.groupby("imageName").size().mean())

    # A1 reproduce vs baked (pooled, native scale)
    recomputed = _recompute(df, dim)
    baked = _pooled_baked(run, dim)
    a1 = {}
    for k in ("mae", "spearman", "mean_bias", "baseline_mae_global_mean",
              "baseline_mae_image_mean"):
        if k in baked and k in recomputed:
            a1[k] = {"recomputed": round(recomputed[k], 5),
                     "baked": round(baked[k], 5),
                     "abs_diff": round(abs(recomputed[k] - baked[k]), 5)}
    max_diff = max((v["abs_diff"] for v in a1.values()), default=None)

    return {
        "run": run.name, "dataset": run.dataset, "mode": run.mode,
        "temperature": run.config.get("temperature"),
        "n_rows": n_rows, "baked_n_ratings": baked_n,
        "row_vs_baked_match": (n_rows == baked_n),
        "duplicate_keys": dup,
        "duplicate_detail": dup_detail,
        "n_users": len(run.users),
        "mean_raters_per_image": round(mean_raters, 2),
        "parse_fail": {"n": n_fail, "frac": round(fail_frac, 5),
                       "images_with_any_fail_frac": round(fail_img_share, 4)},
        "degenerate_pred_frac": round(degen_frac, 4),
        "A1_reproduce": a1,
        "A1_max_abs_diff": None if max_diff is None else round(max_diff, 5),
        "A1_reproduces": (max_diff is not None and max_diff < 1e-3),
    }


def coverage_pairing(runs: dict[str, Run]) -> dict:
    """A4 — do full and blind cover the identical (image,user) task set?"""
    out = {}
    for ds in ("para", "eva", "lapis"):
        f, b = runs.get(f"{ds}_full"), runs.get(f"{ds}_blind")
        if not (f and b):
            continue
        # keys as strings to avoid int/str userId mismatch
        kf = set(map(tuple, f.df[KEY].astype(str).itertuples(index=False, name=None)))
        kb = set(map(tuple, b.df[KEY].astype(str).itertuples(index=False, name=None)))
        out[ds] = {"n_full": len(kf), "n_blind": len(kb),
                   "in_full_only": len(kf - kb), "in_blind_only": len(kb - kf),
                   "shared": len(kf & kb),
                   "identical_task_set": (kf == kb)}
    return out


def main() -> None:
    runs = {name: load_run(name) for name in RUNS}
    per_run = {name: audit_run(r) for name, r in runs.items()}
    pairing = coverage_pairing(runs)
    report = {"per_run": per_run, "full_blind_pairing": pairing,
              "notes": {
                  "A8_baseline_labelling":
                      "baseline_mae_image_mean uses each test image's own raters' mean "
                      "-> it is the empirical GROUP oracle / C1 target, NOT an individual "
                      "floor the model should beat. baseline_mae_global_mean is the true "
                      "individual floor.",
                  "A5_decoding_confound":
                      "full runs are temp=0.0 (greedy), blind runs temp=0.7 -> full/blind "
                      "differences confound persona conditioning with decoding temperature; "
                      "restrict full-vs-blind to rank/mean metrics until a matched-temp rerun.",
              }}
    path = write_json(report, "audit.json")

    # ---- console summary ----
    print(f"\nWrote {path}\n")
    hdr = f"{'run':12}{'rows':>8}{'=baked':>7}{'dupes':>7}{'parseF':>8}{'degen':>7}{'raters':>7}{'A1ok':>6}{'A1Δmax':>8}"
    print(hdr); print("-" * len(hdr))
    for name in RUNS:
        a = per_run[name]
        print(f"{name:12}{a['n_rows']:>8}{str(a['row_vs_baked_match']):>7}"
              f"{a['duplicate_keys']:>7}{a['parse_fail']['frac']:>8.4f}"
              f"{a['degenerate_pred_frac']:>7.2f}{a['mean_raters_per_image']:>7.1f}"
              f"{str(a['A1_reproduces']):>6}{a['A1_max_abs_diff']:>8.4f}")
    print("\nfull/blind identical task set:")
    for ds, p in pairing.items():
        print(f"  {ds:6} identical={p['identical_task_set']}  shared={p['shared']}  "
              f"full_only={p['in_full_only']}  blind_only={p['in_blind_only']}")


if __name__ == "__main__":
    main()
