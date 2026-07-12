"""#1 Multi-dimension extension — the temp-robust analyses on EVERY rated axis.

We had only analyzed the primary axis. This runs, for every secondary dimension, the metrics
that do NOT depend on decoding temperature (ceiling, calibrated group error, persona value):
  - Exp 0 ceiling: ICC(1) single-rating reliability, ICC(k) group-mean reliability
  - calibrated group MAE vs the population-mean prior (does the aggregate beat the prior?)
  - persona value: within-image corr(pred_dev, gt_dev) for full vs blind

This directly tests C1's stated scope (predict difficulty + the EVA attribute votes) and C2's
"structured signals" (PARA contentPreference / willingnessToShare, EVA visual/composition/...).
Normalized [0,1] scale. Pure re-analysis, no inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibration import cross_fit_calibrate, dedup
from common import SCALES, load_run, write_json
from exp0_ceiling import oneway_variance

DATASETS = {"PARA": ("para_full", "para_blind"),
            "EVA": ("eva_full", "eva_blind"),
            "LAPIS": ("lapis_full", "lapis_blind")}


def _within_corr(df: pd.DataFrame, dim: str) -> float:
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = df.dropna(subset=[gt, pred])
    dev_p = d[pred] - d.groupby("imageName")[pred].transform("mean")
    dev_g = d[gt] - d.groupby("imageName")[gt].transform("mean")
    if dev_p.std() < 1e-9 or dev_g.std() < 1e-9:
        return 0.0
    return float(stats.pearsonr(dev_p, dev_g).statistic)


def _dim_row(dataset: str, dim: str, full: pd.DataFrame, blind: pd.DataFrame) -> dict:
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = full.dropna(subset=[gt, pred]).copy()
    dec = oneway_variance(d, gt, "imageName")

    d["cal"] = cross_fit_calibrate(d, dim)
    d = d.dropna(subset=["cal"])
    grp = d.groupby("imageName").agg(pm=("cal", "mean"), om=(gt, "mean"))
    group_mae = float((grp["pm"] - grp["om"]).abs().mean())
    prior = float((grp["om"] - d[gt].mean()).abs().mean())
    grp_rank = float(stats.spearmanr(grp["pm"], grp["om"]).statistic)

    pv = _within_corr(full, dim) - _within_corr(blind, dim)
    return {"dataset": dataset, "dim": dim,
            "ICC1": round(dec["ICC1_single_rating"], 3),
            "ICCk": round(dec["ICCk_group_mean"], 3),
            "group_mae_cal": round(group_mae, 4),
            "pop_prior": round(prior, 4),
            "beats_prior": group_mae < prior,
            "group_rank": round(grp_rank, 3),
            "persona_value": round(pv, 4)}


def main() -> None:
    rows = []
    for ds, (fn, bn) in DATASETS.items():
        full, blind = dedup(load_run(fn).df), dedup(load_run(bn).df)
        for dim in SCALES[ds]:
            if f"{dim}_gt_norm" not in full.columns:
                continue
            rows.append(_dim_row(ds, dim, full, blind))
    write_json(rows, "dims_extended.json")

    print(f"\nWrote results/dims_extended.json\n")
    hdr = (f"{'dataset':7}{'dimension':20}{'ICC1':>6}{'ICCk':>6}"
           f"{'grpMAEcal':>10}{'prior':>8}{'beats':>7}{'grpRank':>9}{'persoVal':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        star = "*" if r["dim"] in ("aestheticScore", "score", "rating") else " "
        print(f"{r['dataset']:6}{star}{r['dim']:20}{r['ICC1']:>6.2f}{r['ICCk']:>6.2f}"
              f"{r['group_mae_cal']:>10.4f}{r['pop_prior']:>8.4f}{str(r['beats_prior']):>7}"
              f"{r['group_rank']:>9.3f}{r['persona_value']:>10.4f}")
    print("\n* = primary axis. beats = calibrated aggregate beats population prior.")
    print("persona_value = within-image corr gain (full - blind); >0 means persona helps.")


if __name__ == "__main__":
    main()
