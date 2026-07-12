"""#1 Rater-level leniency capture — what about a person does the persona capture?

Distinct from attribute-level steerability: does the judge reproduce each rater's OVERALL
harshness/leniency (their mean deviation from the crowd), regardless of attributes? For every
rater we compute real leniency = mean(gt - image_mean) and VLM leniency = mean(pred - image_mean),
then correlate across raters (Spearman, bootstrap CI over raters), for full vs the blind control.
Temp-0 collapse makes this a floor. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibration import dedup
from common import PRIMARY_DIM, load_run, write_json

PAIRS = {"PARA": ("para_full", "para_blind"), "EVA": ("eva_full", "eva_blind"),
         "LAPIS": ("lapis_full", "lapis_blind")}
MIN_RATINGS = 10
RNG = np.random.default_rng(0)


def _leniency(run: str, dim: str) -> pd.DataFrame:
    df = dedup(load_run(run).df)
    g, p = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = df.dropna(subset=[g, p]).copy()
    d["gd"] = d[g] - d.groupby("imageName")[g].transform("mean")
    d["pd"] = d[p] - d.groupby("imageName")[p].transform("mean")
    d["userId"] = d["userId"].astype(str)
    r = d.groupby("userId").agg(real=("gd", "mean"), vlm=("pd", "mean"), n=("gd", "size"))
    return r[r["n"] >= MIN_RATINGS]


def _corr_ci(r: pd.DataFrame):
    rho = float(stats.spearmanr(r["real"], r["vlm"]).statistic)
    real, vlm = r["real"].to_numpy(), r["vlm"].to_numpy()
    boot = []
    for _ in range(1000):
        idx = RNG.integers(0, len(real), len(real))
        boot.append(stats.spearmanr(real[idx], vlm[idx]).statistic)
    return rho, [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


def main() -> None:
    report = {}
    for ds, (full, blind) in PAIRS.items():
        dim = PRIMARY_DIM[ds]
        rf, rb = _leniency(full, dim), _leniency(blind, dim)
        rho_f, ci_f = _corr_ci(rf)
        rho_b, _ = _corr_ci(rb)
        report[ds] = {"n_raters": int(len(rf)), "leniency_corr_full": round(rho_f, 4),
                      "ci95_full": ci_f, "leniency_corr_blind": round(rho_b, 4)}
    write_json(report, "rater_leniency.json")
    print("\nWrote results/rater_leniency.json\n")
    hdr = f"{'dataset':7}{'raters':>8}{'leniency corr (full)':>24}{'blind':>9}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        print(f"{ds:7}{m['n_raters']:>8}"
              f"{str(m['leniency_corr_full'])+' '+str(m['ci95_full']):>24}"
              f"{m['leniency_corr_blind']:>9.3f}")
    print("\ncorr(real rater leniency, VLM-assigned leniency): does the persona capture who is")
    print("globally harsh/lenient? full >> blind(~0) and CI>0 => yes (a floor, temp-0 dampens it).")


if __name__ == "__main__":
    main()
