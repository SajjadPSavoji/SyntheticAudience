"""B1 — steerability gate (research_plan.md sec.5.2). Pure re-analysis, no inference.

Question: does the persona actually steer the frozen judge in the direction the DATA says
it should? If personas barely move the prediction (steerability ~ 0), the persona is
non-functional and the paper pivots to the ceiling finding.

Method (image-controlled, uses the persona-conditioned *_full run only):
  - Image-center predictions and ground truth: subtract each image's mean, leaving each
    rater's DEVIATION from the crowd on that image. This removes shared image quality so we
    isolate the persona/rater effect.
  - For each rater attribute, bin into levels (numeric -> terciles; categorical -> levels
    with >= MIN_CELL ratings). Per level compute:
        empirical effect = mean(gt deviation | level)     # how this group really deviates
        VLM effect       = mean(pred deviation | level)    # how the persona moved the judge
  - Steerability = Pearson corr(VLM effect, empirical effect) across all (attribute,level)
    cells, plus the fraction of cells whose sign matches. Also reported per attribute.

Caveat: `full` ran at temperature 0, so ~half of images have zero prediction spread across
personas (see audit A7); those images contribute no VLM deviation and attenuate the measured
steerability. This is a floor, not a ceiling, on the true persona effect.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from attrs import ATTRS, load_attributes
from common import OUT_DIR, PRIMARY_DIM, ensure_out, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
MIN_CELL = 50      # min ratings for a level to count
N_BINS = 3         # terciles for numeric attributes


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.endswith("_norm")]
    return df.drop_duplicates(subset=cols)


def _levels(series: pd.Series) -> pd.Series:
    """Map an attribute column to discrete level labels (terciles if numeric)."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9 and num.nunique() > 6:
        try:
            return pd.qcut(num, N_BINS, labels=[f"q{i+1}" for i in range(N_BINS)],
                           duplicates="drop").astype(str)
        except ValueError:
            pass
    return series.astype(str)


def steer_dataset(dataset: str, run_name: str) -> dict:
    r = load_run(run_name)
    dim = PRIMARY_DIM[dataset]
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    df = dedup(r.df).dropna(subset=[gt, pred]).copy()
    df["userId"] = df["userId"].astype(str)

    # image-center -> rater deviations
    df["gt_dev"] = df[gt] - df.groupby("imageName")[gt].transform("mean")
    df["pred_dev"] = df[pred] - df.groupby("imageName")[pred].transform("mean")

    attrs = load_attributes(dataset)
    df = df.join(attrs, on="userId")

    cells = []            # (attribute, level, n, emp, vlm)
    per_attr = {}
    for a in ATTRS[dataset]:
        if a not in df.columns:
            continue
        lv = _levels(df[a])
        g = df.assign(_lv=lv).groupby("_lv").agg(
            n=("gt_dev", "size"), emp=("gt_dev", "mean"), vlm=("pred_dev", "mean"))
        g = g[g["n"] >= MIN_CELL]
        if len(g) < 2:
            continue
        for lvl, row in g.iterrows():
            cells.append({"attribute": a, "level": lvl, "n": int(row["n"]),
                          "empirical_effect": float(row["emp"]),
                          "vlm_effect": float(row["vlm"])})
        # per-attribute correlation across its levels (if >=3 levels)
        if len(g) >= 3:
            per_attr[a] = {
                "n_levels": int(len(g)),
                "corr": round(float(stats.pearsonr(g["vlm"], g["emp"]).statistic), 3),
                "sign_agreement": round(float((np.sign(g["vlm"]) == np.sign(g["emp"])).mean()), 3),
            }

    cdf = pd.DataFrame(cells)
    overall_corr = float(stats.pearsonr(cdf["vlm_effect"], cdf["empirical_effect"]).statistic)
    sign_agree = float((np.sign(cdf["vlm_effect"]) == np.sign(cdf["empirical_effect"])).mean())
    # spread of the VLM's persona effect vs the empirical effect (is it even moving?)
    return {
        "dataset": dataset, "primary_dim": dim, "n_cells": int(len(cdf)),
        "steerability_corr": round(overall_corr, 3),
        "sign_agreement_frac": round(sign_agree, 3),
        "vlm_effect_std": round(float(cdf["vlm_effect"].std()), 4),
        "empirical_effect_std": round(float(cdf["empirical_effect"].std()), 4),
        "vlm_over_empirical_amplitude": round(
            float(cdf["vlm_effect"].std() / cdf["empirical_effect"].std()), 3),
        "per_attribute": per_attr,
        "_cells": cells,
    }


def plot(report: dict) -> str:
    figdir = os.path.join(OUT_DIR, "figs")
    os.makedirs(figdir, exist_ok=True)
    dss = [k for k in report if not k.startswith("_")]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 4))
    if len(dss) == 1:
        axes = [axes]
    for ax, ds in zip(axes, dss):
        cdf = pd.DataFrame(report[ds]["_cells"])
        ax.axhline(0, color="#aaa", lw=0.7); ax.axvline(0, color="#aaa", lw=0.7)
        ax.scatter(cdf["empirical_effect"], cdf["vlm_effect"],
                   s=np.sqrt(cdf["n"]), alpha=0.6)
        lim = max(cdf["empirical_effect"].abs().max(), cdf["vlm_effect"].abs().max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "--", color="#c44", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("empirical group deviation")
        ax.set_ylabel("VLM persona deviation")
        ax.set_title(f"{ds}  (r={report[ds]['steerability_corr']}, "
                     f"sign={report[ds]['sign_agreement_frac']})")
        ax.grid(True, alpha=0.3)
    fig.suptitle("B1 — steerability: does the persona move the judge the way the data says?")
    fig.tight_layout()
    path = os.path.join(figdir, "b1_steerability.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ensure_out()
    report = {ds: steer_dataset(ds, run) for ds, run in DATASETS.items()}
    fig = plot(report)
    # strip bulky cell lists from the saved json summary but keep them for the figure
    slim = {ds: {k: v for k, v in m.items() if k != "_cells"} for ds, m in report.items()}
    slim["_figure"] = fig
    path = write_json(slim, "steerability.json")

    print(f"\nWrote {path}\nWrote {fig}\n")
    hdr = f"{'dataset':7}{'cells':>7}{'steer r':>9}{'signAgree':>11}{'vlmAmp/emp':>12}"
    print(hdr); print("-" * len(hdr))
    for ds in DATASETS:
        m = report[ds]
        print(f"{ds:7}{m['n_cells']:>7}{m['steerability_corr']:>9.3f}"
              f"{m['sign_agreement_frac']:>11.3f}{m['vlm_over_empirical_amplitude']:>12.3f}")
    print("\nsteer r = corr(VLM persona effect, real group effect) across attribute levels.")
    print("signAgree = fraction of levels the VLM moves in the correct direction.")
    print("vlmAmp/emp = how big the VLM's persona swing is vs the real one (<1 = under-moves).")
    print("GATE (plan sec.5.2): steer r > 0 -> persona functional; ~0 -> pivot to ceiling finding.")
    print("Caveat: full is temp 0 -> ~half of images have zero persona spread; this is a floor.")


if __name__ == "__main__":
    main()
