"""Tier1 #3 — clean full-vs-blind persona value (decoding-invariant).

The full-vs-blind MAE comparison is confounded by temperature (full 0.0, blind 0.7). But a
rank/within-image comparison is decoding-robust and isolates what the persona actually buys:

  within-image signal = corr(pred_dev, gt_dev), where *_dev is the rating minus its image mean.

Both runs share the same image, so the between-image quality signal is common; the WITHIN-image
deviation is where a persona can help. For the persona-blind run there is no persona, so its
within-image prediction variation is only temp-0.7 sampling noise and should correlate ~0 with
the real rater deviation. If personas work, `full` should show within-image corr > blind.

Reported per dataset: between-image rank (shared) and within-image corr for full vs blind,
with 1000x bootstrap CIs clustered by image. Exact-duplicate rows dropped first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import PRIMARY_DIM, load_run, write_json

PAIRS = {"PARA": ("para_full", "para_blind"),
         "EVA": ("eva_full", "eva_blind"),
         "LAPIS": ("lapis_full", "lapis_blind")}
RNG = np.random.default_rng(0)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.endswith("_norm")]
    return df.drop_duplicates(subset=cols)


def _signals(run_name: str, dim: str) -> dict:
    r = load_run(run_name)
    gt, pred = f"{dim}_gt_norm", f"{dim}_pred_norm"
    d = dedup(r.df).dropna(subset=[gt, pred]).copy()
    d["gt_dev"] = d[gt] - d.groupby("imageName")[gt].transform("mean")
    d["pred_dev"] = d[pred] - d.groupby("imageName")[pred].transform("mean")

    # between-image (shared) rank
    img = d.groupby("imageName").agg(pm=(pred, "mean"), om=(gt, "mean"))
    between = float(stats.spearmanr(img["pm"], img["om"]).statistic)

    # within-image persona signal (+ bootstrap over images)
    within = float(stats.pearsonr(d["pred_dev"], d["gt_dev"]).statistic)
    imgs = np.array(d["imageName"].unique(), dtype=object)
    by_img = {k: g[["pred_dev", "gt_dev"]].to_numpy() for k, g in d.groupby("imageName")}
    boot = []
    for _ in range(1000):
        pick = imgs[RNG.integers(0, len(imgs), len(imgs))]
        arr = np.concatenate([by_img[i] for i in pick])
        if arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
            boot.append(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    # how much within-image prediction spread even exists (temp/persona-driven)
    frac_moving = float((d.groupby("imageName")["pred_dev"].std() > 1e-9).mean())
    return {"between_image_rank": round(between, 4),
            "within_image_corr": round(within, 4),
            "within_image_corr_ci95": ci,
            "frac_images_with_pred_spread": round(frac_moving, 4)}


def main() -> None:
    report = {}
    for ds, (full, blind) in PAIRS.items():
        dim = PRIMARY_DIM[ds]
        f = _signals(full, dim)
        b = _signals(blind, dim)
        report[ds] = {
            "primary_dim": dim,
            "full": f, "blind": b,
            "persona_value_within_image": round(f["within_image_corr"] - b["within_image_corr"], 4),
        }
    path = write_json(report, "persona_value.json")
    print(f"\nWrote {path}\n")
    hdr = (f"{'dataset':7}{'betwRank F/B':>16}{'within F':>10}{'within B':>10}"
           f"{'persona val':>13}{'full CI95':>20}")
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        f, b = m["full"], m["blind"]
        print(f"{ds:7}{f['between_image_rank']:>8.3f}/{b['between_image_rank']:<7.3f}"
              f"{f['within_image_corr']:>10.3f}{b['within_image_corr']:>10.3f}"
              f"{m['persona_value_within_image']:>13.3f}"
              f"{str(f['within_image_corr_ci95']):>20}")
    print("\nbetwRank = between-image rank (shared, ~equal for full/blind, as expected).")
    print("within F/B = corr(pred_dev, gt_dev): the persona's individual signal (full) vs the")
    print("no-persona floor (blind, ~0). persona val = full - blind within-image gain.")
    print("Note: full is temp 0 -> personas drive the whole within-image spread; blind's spread")
    print("is temp-0.7 sampling noise. A matched-temp rerun would tighten this contrast.")


if __name__ == "__main__":
    main()
