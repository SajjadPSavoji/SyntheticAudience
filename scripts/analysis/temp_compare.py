"""Temp-0 vs temp-0.7 comparison (§14.15). Did the re-run lift the temp-0 floors?

Runs the temperature-sensitive metrics on the persona (`full`) runs at temp 0 vs temp 0.7, for
PARA and EVA (no LAPIS temp-0.7 run yet). Reuses the existing analysis functions. Pure re-analysis.

NOTE: PARA temp-0.7 is ALSO wider (4000 images vs 2000) — its column mixes the temperature and
coverage changes; EVA temp-0.7 is the same task set, so it is a clean temperature-only comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from c1_separation import SLICES, _gap_cells, _prep, _separation
from c2_ncurve import individual_vs_group, n_curve
from calibration import dedup
from common import PRIMARY_DIM, load_run, write_json
from response_style import analyze as style_analyze
from steerability import steer_dataset

RUNS = {"PARA": ("para_full", "para_full_t07"), "EVA": ("eva_full", "eva_full_t07"),
        "LAPIS": ("lapis_full", "lapis_full_t07")}


def _degen_and_spread(df: pd.DataFrame, dim: str) -> dict:
    pc = f"pred_{dim}"
    g = df.groupby("imageName")[pc]
    return {"degenerate_frac": round(float((g.nunique() <= 1).mean()), 3),
            "mean_within_image_pred_std": round(float(g.std().mean()), 4)}


def _persona_value(run: str, blind_run: str, dim: str) -> float:
    def wc(name):
        d = dedup(load_run(name).df).dropna(subset=[f"{dim}_gt_norm", f"{dim}_pred_norm"])
        dp = d[f"{dim}_pred_norm"] - d.groupby("imageName")[f"{dim}_pred_norm"].transform("mean")
        dg = d[f"{dim}_gt_norm"] - d.groupby("imageName")[f"{dim}_gt_norm"].transform("mean")
        return 0.0 if dp.std() < 1e-9 else float(stats.pearsonr(dp, dg).statistic)
    return round(wc(run) - wc(blind_run), 4)


def _c1_overall(run: str, dataset: str) -> dict:
    df = _prep(run, dataset)
    cells = pd.concat([_gap_cells(df, a) for a in SLICES[dataset] if a in df.columns],
                      ignore_index=True)
    return _separation(cells)


def compare(dataset: str) -> dict:
    t0, t07 = RUNS[dataset]
    dim = PRIMARY_DIM[dataset]
    blind = f"{dataset.lower()}_blind"
    out = {"dataset": dataset, "primary_dim": dim}
    for label, run in [("temp0", t0), ("temp07", t07)]:
        df = dedup(load_run(run).df)
        gap = individual_vs_group(df, dim)
        curve = {c["N"]: c["group_mae"] for c in n_curve(df, dim)}
        out[label] = {
            "n_images": int(df["imageName"].nunique()),
            "n_ratings": int(len(df)),
            **_degen_and_spread(df, dim),
            "ncurve_N1": curve.get(1), "ncurve_N5": curve.get(5), "ncurve_N20": curve.get(20),
            "ncurve_drop_N1_to_N20": round((curve.get(1, 0) - curve.get(20, 0)), 4),
            "group_mae": gap["group"]["mae"],
            "c1_separation": _c1_overall(run, dataset),
            "steerability": steer_dataset(dataset, run)["steerability_corr"],
            "persona_value": _persona_value(run, blind, dim),
            "scale_usage_ratio": style_analyze(dataset, run)["scale_usage_ratio"],
        }
    return out


def main() -> None:
    report = {ds: compare(ds) for ds in RUNS}
    write_json(report, "temp_compare.json")
    print("\nWrote results/temp_compare.json\n")
    metrics = [("degenerate_frac", "degen frac"), ("mean_within_image_pred_std", "within-img std"),
               ("ncurve_drop_N1_to_N20", "N-curve drop N1→N20"), ("c1_separation", "C1 separation"),
               ("steerability", "steerability r"), ("persona_value", "persona value"),
               ("scale_usage_ratio", "scale usage ratio")]
    for ds, m in report.items():
        print(f"### {ds}  (temp0 imgs={m['temp0']['n_images']}, temp0.7 imgs={m['temp07']['n_images']})")
        print(f"{'metric':22}{'temp 0':>12}{'temp 0.7':>12}")
        print("-" * 46)
        for key, lab in metrics:
            v0, v7 = m["temp0"][key], m["temp07"][key]
            if key == "c1_separation":
                v0 = v0["corr"]; v7 = v7["corr"]
            print(f"{lab:22}{v0:>12}{v7:>12}")
        print()
    print("Read: if temp 0.7 lifted the floor, expect degen frac DOWN, within-img std UP,")
    print("N-curve drop UP (aggregation now cancels variance), separation/steerability UP.")


if __name__ == "__main__":
    main()
