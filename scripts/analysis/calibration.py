"""B4 — post-hoc calibration (research_plan.md sec.5.1 step 4, sec.8.2).

The Exp0/C2 results showed the dominant error is a large positive BIAS the model applies to
every rating; aggregation cannot remove it. Post-hoc calibration maps raw -> calibrated
scores (isotonic; monotonic, so it changes MAE/bias but NOT rank), fit on a held-out split
and applied out-of-fold, touching no weights.

We use 2-fold cross-fitting by IMAGE (disjoint image folds, seed 0) so every rating is
scored by a calibrator that never saw its image, and all data is still evaluated. Reported
raw-vs-calibrated on the primary dim, normalized [0,1] scale:
  individual MAE / bias / Spearman, group-mean MAE, and the population-mean prior.

The decisive question: once bias is removed, does the VLM's aggregate (which already tracks
between-image order at rho~0.7) beat the population-mean prior it lost to uncalibrated?
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from common import OUT_DIR, PRIMARY_DIM, ensure_out, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
RNG = np.random.default_rng(0)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.endswith("_norm")]
    return df.drop_duplicates(subset=cols)


def cross_fit_calibrate(df: pd.DataFrame, dim: str, k: int = 2) -> pd.Series:
    """Return out-of-fold isotonic-calibrated predictions aligned to df.index."""
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = df.dropna(subset=[gt, pred])
    images = np.array(d["imageName"].unique(), dtype=object)
    RNG.shuffle(images)
    fold_of = {img: i % k for i, img in enumerate(images)}
    fold = d["imageName"].map(fold_of)
    cal = pd.Series(np.nan, index=d.index)
    for f in range(k):
        fit = d[fold != f]
        ev = d[fold == f]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(fit[pred].to_numpy(), fit[gt].to_numpy())
        cal.loc[ev.index] = iso.predict(ev[pred].to_numpy())
    return cal


def _metrics(d: pd.DataFrame, gt: str, predcol: str) -> dict:
    ind_err = (d[predcol] - d[gt]).abs()
    grp = d.groupby("imageName").agg(pm=(predcol, "mean"), om=(gt, "mean"))
    grp_err = (grp["pm"] - grp["om"]).abs()
    pop_err = (grp["om"] - d[gt].mean()).abs()
    return {
        "individual_mae": round(float(ind_err.mean()), 4),
        "individual_bias": round(float((d[predcol] - d[gt]).mean()), 4),
        "spearman": round(float(stats.spearmanr(d[predcol], d[gt]).statistic), 4),
        "group_mae": round(float(grp_err.mean()), 4),
        "group_rank": round(float(stats.spearmanr(grp["pm"], grp["om"]).statistic), 4),
        "population_prior_group_mae": round(float(pop_err.mean()), 4),
    }


def analyze(dataset: str, run_name: str) -> dict:
    r = load_run(run_name)
    dim = PRIMARY_DIM[dataset]
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    df = dedup(r.df).dropna(subset=[gt, pred]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    raw = _metrics(df, gt, pred)
    cal = _metrics(df, gt, "cal")
    return {"dataset": dataset, "primary_dim": dim, "n": int(len(df)),
            "raw": raw, "calibrated": cal,
            "beats_pop_prior_raw": raw["group_mae"] < raw["population_prior_group_mae"],
            "beats_pop_prior_calibrated": cal["group_mae"] < cal["population_prior_group_mae"]}


def plot(report: dict) -> str:
    figdir = os.path.join(OUT_DIR, "figs")
    os.makedirs(figdir, exist_ok=True)
    dss = [k for k in report if not k.startswith("_")]
    fig, axes = plt.subplots(1, len(dss), figsize=(4 * len(dss), 4), sharey=False)
    if len(dss) == 1:
        axes = [axes]
    for ax, ds in zip(axes, dss):
        m = report[ds]
        labels = ["indiv\nraw", "indiv\ncal", "group\nraw", "group\ncal", "pop\nprior"]
        vals = [m["raw"]["individual_mae"], m["calibrated"]["individual_mae"],
                m["raw"]["group_mae"], m["calibrated"]["group_mae"],
                m["calibrated"]["population_prior_group_mae"]]
        colors = ["#c44", "#4a4", "#c44", "#4a4", "#48c"]
        ax.bar(labels, vals, color=colors)
        ax.set_title(ds)
        ax.set_ylabel("MAE (normalized)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("B4 — effect of isotonic calibration on MAE (red=raw, green=calibrated)")
    fig.tight_layout()
    path = os.path.join(figdir, "b4_calibration.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ensure_out()
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    report["_figure"] = plot(report)
    path = write_json(report, "calibration.json")
    print(f"\nWrote {path}\nWrote {report['_figure']}\n")
    hdr = (f"{'dataset':7}{'indivMAE raw→cal':>20}{'grpMAE raw→cal':>20}"
           f"{'popPrior':>10}{'spear':>7}{'beatsPrior(cal)':>16}")
    print(hdr); print("-" * len(hdr))
    for ds in DATASETS:
        m = report[ds]
        print(f"{ds:7}"
              f"{m['raw']['individual_mae']:>10.4f}→{m['calibrated']['individual_mae']:<9.4f}"
              f"{m['raw']['group_mae']:>10.4f}→{m['calibrated']['group_mae']:<9.4f}"
              f"{m['calibrated']['population_prior_group_mae']:>10.4f}"
              f"{m['calibrated']['spearman']:>7.3f}"
              f"{str(m['beats_pop_prior_calibrated']):>16}")
    print("\nSpearman barely moves (isotonic is monotonic within each fold; the small drop is")
    print("from calibration ties + the two folds' maps differing) — a sanity check on rank.")
    print("The test: does calibrated group MAE beat the population prior (rightmost)?")


if __name__ == "__main__":
    main()
