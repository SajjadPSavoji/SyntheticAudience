"""Tier1 #1 — C1 between-group separation (the headline test). Pure re-analysis, no inference.

The decisive C1 evidence (proposal sec.2, plan sec.8.1) is NOT matching the grand mean but
capturing how groups DIVERGE on the same image: do art experts and novices differ on a given
painting in the direction the data shows?

Method (image-controlled, on calibrated predictions):
  - Slice raters by an attribute (levels). For each (image, level) with >= MIN_CELL raters,
    take the observed mean rating and the predicted mean rating.
  - For every pair of levels co-present on an image, form the observed gap and predicted gap.
  - Between-group separation = corr(predicted gaps, observed gaps) across all (image, pair)
    cells, with 1000x bootstrap CIs clustered by image. Computed for the persona run (`full`)
    and the no-persona control (`blind`, expected ~0).
  - Secondary: per-slice distribution match (Wasserstein-1, pooled per level) for full vs
    blind vs the population-mean prior.

Predictions are isotonic-calibrated out-of-fold (reusing calibration.cross_fit_calibrate).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

from calibration import cross_fit_calibrate, dedup
from common import OUT_DIR, PRIMARY_DIM, ensure_out, load_run, write_json
from steerability import _levels

SLICES = {
    "PARA": ["artExperience", "photographyExperience", "age"],
    "EVA": ["photographic_level", "age"],
    "LAPIS": ["nationality", "art_interest", "age"],
}
from attrs import load_attributes

MIN_CELL = 4      # min raters of a level on an image to estimate its mean
RNG = np.random.default_rng(0)


def _prep(run_name: str, dataset: str) -> pd.DataFrame:
    r = load_run(run_name)
    dim = PRIMARY_DIM[dataset]
    gt = f"{dim}_gt_norm"
    df = dedup(r.df).dropna(subset=[gt, f"{dim}_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    df["userId"] = df["userId"].astype(str)
    df = df.join(load_attributes(dataset), on="userId")
    df["gt"] = df[gt]
    return df


def _gap_cells(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    """All (image, level-pair) observed/predicted gap cells for one slicing attribute."""
    d = df.assign(_lv=_levels(df[attr]))
    d = d[d["_lv"].notna() & (d["_lv"].astype(str) != "nan")]
    cell = (d.groupby(["imageName", "_lv"])
              .agg(n=("gt", "size"), obs=("gt", "mean"), pred=("cal", "mean"))
              .reset_index())
    cell = cell[cell["n"] >= MIN_CELL]
    rows = []
    for img, sub in cell.groupby("imageName"):
        v = sub[["obs", "pred"]].to_numpy()
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                rows.append((img, v[i, 0] - v[j, 0], v[i, 1] - v[j, 1]))
    return pd.DataFrame(rows, columns=["imageName", "obs_gap", "pred_gap"])


def _separation(cells: pd.DataFrame) -> dict:
    if len(cells) < 10:
        return {"n_cells": int(len(cells)), "corr": None, "sign_agreement": None,
                "ci95": [None, None]}
    corr = float(stats.pearsonr(cells["pred_gap"], cells["obs_gap"]).statistic)
    sign = float((np.sign(cells["pred_gap"]) == np.sign(cells["obs_gap"])).mean())
    # bootstrap clustered by image
    by_img = {k: g[["obs_gap", "pred_gap"]].to_numpy() for k, g in cells.groupby("imageName")}
    imgs = np.array(list(by_img), dtype=object)
    boot = []
    for _ in range(1000):
        pick = imgs[RNG.integers(0, len(imgs), len(imgs))]
        arr = np.concatenate([by_img[i] for i in pick])
        if arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
            boot.append(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    return {"n_cells": int(len(cells)), "corr": round(corr, 4),
            "sign_agreement": round(sign, 4), "ci95": ci}


def _dist_match(df: pd.DataFrame, attr: str) -> dict:
    """Per-slice Wasserstein-1 (pooled per level): full/blind-style pred vs obs, and prior."""
    d = df.assign(_lv=_levels(df[attr]))
    d = d[d["_lv"].notna() & (d["_lv"].astype(str) != "nan")]
    glob = float(d["gt"].mean())
    w_pred, w_prior = [], []
    for lv, g in d.groupby("_lv"):
        if len(g) < 30:
            continue
        obs = g["gt"].to_numpy()
        w_pred.append(wasserstein_distance(g["cal"].to_numpy(), obs))
        w_prior.append(wasserstein_distance(np.full(len(obs), glob), obs))
    if not w_pred:
        return {"n_slices": 0}
    return {"n_slices": len(w_pred),
            "mean_W1_pred": round(float(np.mean(w_pred)), 4),
            "mean_W1_population_prior": round(float(np.mean(w_prior)), 4)}


def analyze(dataset: str) -> dict:
    full = _prep(f"{dataset.lower()}_full", dataset)
    blind = _prep(f"{dataset.lower()}_blind", dataset)
    out = {"dataset": dataset, "primary_dim": PRIMARY_DIM[dataset], "by_attribute": {}}
    all_full, all_blind = [], []
    for a in SLICES[dataset]:
        if a not in full.columns:
            continue
        cf, cb = _gap_cells(full, a), _gap_cells(blind, a)
        all_full.append(cf.assign(attr=a)); all_blind.append(cb.assign(attr=a))
        out["by_attribute"][a] = {
            "full_separation": _separation(cf),
            "blind_separation": _separation(cb),
            "distribution_match_full": _dist_match(full, a),
        }
    # pooled over all attributes
    out["overall"] = {
        "full_separation": _separation(pd.concat(all_full, ignore_index=True)),
        "blind_separation": _separation(pd.concat(all_blind, ignore_index=True)),
    }
    return out


def plot(report: dict) -> str:
    figdir = os.path.join(OUT_DIR, "figs")
    os.makedirs(figdir, exist_ok=True)
    dss = [k for k in report if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(dss)); w = 0.35
    full = [report[d]["overall"]["full_separation"]["corr"] or 0 for d in dss]
    blind = [report[d]["overall"]["blind_separation"]["corr"] or 0 for d in dss]
    f_err = np.array([[f - report[d]["overall"]["full_separation"]["ci95"][0],
                       report[d]["overall"]["full_separation"]["ci95"][1] - f]
                      for f, d in zip(full, dss)]).T
    ax.bar(x - w/2, full, w, label="persona (full)", yerr=f_err, capsize=4, color="#4a7")
    ax.bar(x + w/2, blind, w, label="no-persona (blind)", color="#c77")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(dss)
    ax.set_ylabel("between-group separation  corr(pred gap, obs gap)")
    ax.set_title("C1 — between-group separation (calibrated, pooled over slices)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figdir, "c1_separation.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main() -> None:
    ensure_out()
    report = {ds: analyze(ds) for ds in SLICES}
    report["_figure"] = plot(report)
    path = write_json(report, "c1_separation.json")
    print(f"\nWrote {path}\nWrote {report['_figure']}\n")

    hdr = f"{'dataset':7}{'attribute':22}{'full sep (CI95)':>26}{'blind sep':>11}{'cells':>7}"
    print(hdr); print("-" * len(hdr))
    for ds in SLICES:
        for a, m in report[ds]["by_attribute"].items():
            fs, bs = m["full_separation"], m["blind_separation"]
            ci = fs["ci95"]
            cistr = f"{fs['corr']} [{ci[0]},{ci[1]}]" if fs["corr"] is not None else "n/a"
            bstr = f"{bs['corr']}" if bs["corr"] is not None else "n/a"
            print(f"{ds:7}{a:22}{cistr:>26}{bstr:>11}{fs['n_cells']:>7}")
        ov = report[ds]["overall"]
        f, b = ov["full_separation"], ov["blind_separation"]
        print(f"{ds:7}{'OVERALL (pooled)':22}"
              f"{str(f['corr'])+' '+str(f['ci95']):>26}{str(b['corr']):>11}{f['n_cells']:>7}")
        print("-" * len(hdr))
    print("\nfull sep = corr(predicted between-group gap, observed gap), image-controlled.")
    print("CI excludes 0 AND full >> blind (~0) => the model reproduces group DIFFERENCES,")
    print("not just the grand mean. This is the decisive C1 evidence.")


if __name__ == "__main__":
    main()
