"""#3 Bias / fairness diagnostics — per-subgroup calibration error (ethics appendix).

After calibration, is the residual error even across demographic subgroups, or does the judge
systematically over/under-rate for some groups? For each rater attribute we bin into levels and
report the per-level signed error (bias) and MAE on the calibrated primary-dim prediction; the
"fairness gap" is the spread (max - min) across a level set. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attrs import ATTRS, load_attributes
from calibration import cross_fit_calibrate, dedup
from common import PRIMARY_DIM, load_run, write_json
from steerability import _levels

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
MIN_CELL = 100


def analyze(dataset: str, run: str) -> dict:
    r = load_run(run)
    dim = PRIMARY_DIM[dataset]
    gt = f"{dim}_gt_norm"
    df = dedup(r.df).dropna(subset=[gt, f"{dim}_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    df["err"] = df["cal"] - df[gt]
    df["abserr"] = df["err"].abs()
    df["userId"] = df["userId"].astype(str)
    df = df.join(load_attributes(dataset), on="userId")

    out = {}
    worst = {"attribute": None, "mae_gap": 0.0}
    for a in ATTRS[dataset]:
        if a not in df.columns:
            continue
        lv = _levels(df[a])
        g = df.assign(_lv=lv).groupby("_lv").agg(
            n=("err", "size"), bias=("err", "mean"), mae=("abserr", "mean"))
        g = g[g["n"] >= MIN_CELL]
        if len(g) < 2:
            continue
        levels = {str(k): {"n": int(v["n"]), "bias": round(float(v["bias"]), 4),
                           "mae": round(float(v["mae"]), 4)} for k, v in g.iterrows()}
        mae_gap = float(g["mae"].max() - g["mae"].min())
        bias_gap = float(g["bias"].max() - g["bias"].min())
        out[a] = {"levels": levels, "mae_gap": round(mae_gap, 4),
                  "bias_gap": round(bias_gap, 4)}
        if mae_gap > worst["mae_gap"]:
            worst = {"attribute": a, "mae_gap": round(mae_gap, 4)}
    return {"dataset": dataset, "primary_dim": dim, "by_attribute": out,
            "worst_mae_gap": worst}


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    write_json(report, "bias.json")
    print("\nWrote results/bias.json\n")
    print("Per-subgroup calibration MAE gap (max-min across levels) and bias gap:")
    hdr = f"{'dataset':7}{'attribute':22}{'MAE gap':>9}{'bias gap':>10}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        for a, v in m["by_attribute"].items():
            print(f"{ds:7}{a:22}{v['mae_gap']:>9.4f}{v['bias_gap']:>10.4f}")
    print("\nSmall gaps = calibrated judge is fair across that group; large gaps flag subgroups")
    print("the judge systematically mis-rates (report in the ethics appendix).")


if __name__ == "__main__":
    main()
