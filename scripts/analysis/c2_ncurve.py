"""C2 — why aggregation works (research_plan.md sec.8.2). Pure re-analysis, no inference.

Two things, using the persona-conditioned (*_full) predictions only:

 1. Aggregate-vs-individual gap. The VLM's per-rater error (individual) placed next to
    its group-mean error (aggregate over an image's personas vs the observed group mean).
    If aggregation helps, the group error is far smaller than the individual error.

 2. N-personas fidelity curve. For panel sizes N in {1,2,5,10,20,...}, subsample N of an
    image's personas, average their predictions, and measure the error of that panel mean
    against the observed group mean. Expect a monotone decrease that saturates; the
    saturation floor is the VLM's (uncalibrated) bias, which aggregation cannot remove.

All errors are on the normalized [0,1] scale. CIs are 1000x bootstrap over images
(group metrics) / over ratings (individual metrics). Exact-duplicate rows dropped first.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT_DIR, PRIMARY_DIM, ensure_out, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
N_GRID = [1, 2, 5, 10, 20, 50]
N_BOOT = 1000
RNG = np.random.default_rng(0)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.endswith("_norm")]
    return df.drop_duplicates(subset=cols)


def _boot_ci(vals: np.ndarray, stat=np.mean, n=N_BOOT) -> list[float]:
    vals = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    draws = stat(vals[idx], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def individual_vs_group(df: pd.DataFrame, dim: str) -> dict:
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = df.dropna(subset=[gt, pred])

    # individual: per-rating absolute error + rank corr
    ind_err = (d[pred] - d[gt]).abs().to_numpy()
    ind_mae = float(ind_err.mean())
    ind_spear = float(stats.spearmanr(d[pred], d[gt]).statistic)

    # group: per-image predicted mean vs observed mean
    grp = d.groupby("imageName").agg(pred_mean=(pred, "mean"),
                                     obs_mean=(gt, "mean"), n=(gt, "size"))
    grp_err = (grp["pred_mean"] - grp["obs_mean"]).abs().to_numpy()
    grp_mae = float(grp_err.mean())
    grp_bias = float((grp["pred_mean"] - grp["obs_mean"]).mean())
    # does the VLM's group mean track the real between-image differences?
    grp_spear = float(stats.spearmanr(grp["pred_mean"], grp["obs_mean"]).statistic)
    # population-mean prior: predict global mean for every image
    pop_err = (grp["obs_mean"] - d[gt].mean()).abs().to_numpy()

    return {
        "individual": {"mae": round(ind_mae, 4), "spearman": round(ind_spear, 4),
                       "mae_ci95": [round(x, 4) for x in _boot_ci(ind_err)]},
        "group": {"mae": round(grp_mae, 4), "bias": round(grp_bias, 4),
                  "spearman_vs_obs": round(grp_spear, 4),
                  "mae_ci95": [round(x, 4) for x in _boot_ci(grp_err)],
                  "n_images": int(len(grp))},
        "population_mean_prior": {"group_mae": round(float(pop_err.mean()), 4)},
        "aggregate_vs_individual_gap": round(ind_mae - grp_mae, 4),
        "gap_ratio_individual_over_group": round(ind_mae / grp_mae, 2) if grp_mae else None,
    }


def n_curve(df: pd.DataFrame, dim: str, draws_per_image: int = 40) -> list[dict]:
    """Panel-size fidelity curve: error of an N-persona panel mean vs observed group mean."""
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = df.dropna(subset=[gt, pred])
    # per image: array of persona predictions, and the observed group mean
    groups = {img: (g[pred].to_numpy(), g[gt].mean())
              for img, g in d.groupby("imageName")}
    out = []
    for N in N_GRID:
        per_img_err = []
        for preds, obs_mean in groups.values():
            if len(preds) < N:
                continue
            # average over several random N-subsets to reduce sampling noise
            errs = []
            for _ in range(draws_per_image):
                sample = RNG.choice(preds, size=N, replace=False)
                errs.append(abs(sample.mean() - obs_mean))
            per_img_err.append(np.mean(errs))
        per_img_err = np.asarray(per_img_err)
        if len(per_img_err) == 0:
            continue
        out.append({"N": N, "n_images": int(len(per_img_err)),
                    "group_mae": round(float(per_img_err.mean()), 4),
                    "group_mae_ci95": [round(x, 4) for x in _boot_ci(per_img_err)]})
    return out


def plot_curves(curves: dict[str, list[dict]]) -> str:
    figdir = os.path.join(OUT_DIR, "figs")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds, curve in curves.items():
        Ns = [c["N"] for c in curve]
        mae = [c["group_mae"] for c in curve]
        lo = [c["group_mae_ci95"][0] for c in curve]
        hi = [c["group_mae_ci95"][1] for c in curve]
        ax.plot(Ns, mae, marker="o", label=ds)
        ax.fill_between(Ns, lo, hi, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("panel size N (personas)")
    ax.set_ylabel("group-mean error (normalized MAE)")
    ax.set_title("C2 — N-personas fidelity curve (uncalibrated)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figdir, "c2_ncurve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ensure_out()
    report, curves = {}, {}
    for ds, run in DATASETS.items():
        r = load_run(run)
        dim = PRIMARY_DIM[ds]
        df = dedup(r.df)
        gap = individual_vs_group(df, dim)
        curve = n_curve(df, dim)
        curves[ds] = curve
        report[ds] = {"primary_dim": dim, "scale": "normalized [0,1]",
                      "gap": gap, "n_curve": curve}
    fig_path = plot_curves(curves)
    report["_figure"] = fig_path
    path = write_json(report, "c2.json")

    print(f"\nWrote {path}\nWrote {fig_path}\n")
    hdr = f"{'dataset':8}{'indivMAE':>10}{'groupMAE':>10}{'gap x':>8}{'grpRank':>9}{'popPrior':>10}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        if ds.startswith("_"):
            continue
        g = m["gap"]
        print(f"{ds:8}{g['individual']['mae']:>10.4f}{g['group']['mae']:>10.4f}"
              f"{g['gap_ratio_individual_over_group']:>8.2f}"
              f"{g['group']['spearman_vs_obs']:>9.3f}"
              f"{g['population_mean_prior']['group_mae']:>10.4f}")
    print("\nN-curve (group MAE by panel size):")
    for ds in DATASETS:
        pts = ", ".join(f"N{c['N']}={c['group_mae']:.4f}" for c in curves[ds])
        print(f"  {ds:6} {pts}")
    print("\ngap x = individual MAE / group MAE (how much aggregation buys).")
    print("grpRank = Spearman of predicted vs observed image means (between-image signal).")
    print("Note: uncalibrated -> N-curve floor is the VLM bias; see B4 (calibration).")


if __name__ == "__main__":
    main()
