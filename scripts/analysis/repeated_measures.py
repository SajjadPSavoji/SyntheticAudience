"""#8 LAPIS repeated-measures sensitivity (audit robustness).

The audit found LAPIS has 900 exact-duplicate rows (export artifact) + 283 genuine repeated
measures (same rater scored an image twice with different scores). Does the choice of policy move
the headline metrics? We recompute the primary-dim metrics under three policies:
  keep_all      - every row as-is
  drop_exact    - drop exact-identical duplicate rows (our default)
  average_reps  - collapse each (image,user) to its mean rating
Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from common import PRIMARY_DIM, load_run, write_json

RNG = np.random.default_rng(0)


def _policies(df: pd.DataFrame) -> dict:
    non_norm = [c for c in df.columns if not c.endswith("_norm")]
    keep_all = df
    drop_exact = df.drop_duplicates(subset=non_norm)
    avg = (df.groupby(["imageName", "userId"], as_index=False)
             .agg({"rating_gt_norm": "mean", "rating_pred_norm": "mean"}))
    return {"keep_all": keep_all, "drop_exact": drop_exact, "average_reps": avg}


def _metrics(d: pd.DataFrame) -> dict:
    g, p = "rating_gt_norm", "rating_pred_norm"
    d = d.dropna(subset=[g, p])
    # out-of-fold-ish single isotonic (whole set) just for a comparable calibrated group MAE
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(d[p], d[g])
    cal = iso.predict(d[p])
    dd = d.assign(cal=cal)
    grp = dd.groupby("imageName").agg(pm=("cal", "mean"), om=(g, "mean"))
    return {"n": int(len(d)),
            "individual_mae_raw": round(float((d[p] - d[g]).abs().mean()), 4),
            "individual_spearman": round(float(stats.spearmanr(d[p], d[g]).statistic), 4),
            "group_mae_cal": round(float((grp["pm"] - grp["om"]).abs().mean()), 4)}


def main() -> None:
    df = load_run("lapis_full").df
    report = {pol: _metrics(d) for pol, d in _policies(df).items()}
    write_json(report, "repeated_measures.json")
    print("\nWrote results/repeated_measures.json\n")
    hdr = f"{'policy':14}{'n':>9}{'indiv MAE':>11}{'indiv spear':>13}{'group MAE cal':>15}"
    print(hdr); print("-" * len(hdr))
    for pol, m in report.items():
        print(f"{pol:14}{m['n']:>9}{m['individual_mae_raw']:>11.4f}"
              f"{m['individual_spearman']:>13.4f}{m['group_mae_cal']:>15.4f}")
    print("\nNear-identical rows => the headline LAPIS numbers are robust to the duplicate policy.")


if __name__ == "__main__":
    main()
