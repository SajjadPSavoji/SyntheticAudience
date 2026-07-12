"""#4 Content-category performance — where does the audience model work best?

Per image-content category, the calibrated group error and between-image rank. PARA uses its
'semantic' content label (portrait/scene/animal/food/...); EVA uses image_content_category.csv
(numeric category id). Shows which subject matter the judge predicts well. Pure re-analysis.
"""
from __future__ import annotations

import os

import pandas as pd
from scipy import stats

from calibration import cross_fit_calibrate, dedup
from common import PRIMARY_DIM, REPO, load_run, write_json


def para_categories() -> pd.Series:
    p = os.path.join(REPO, "data", "para", "annotation", "PARA-Images.csv")
    d = pd.read_csv(p, usecols=["imageName", "semantic"]).drop_duplicates("imageName")
    return d.set_index("imageName")["semantic"]


def eva_categories() -> pd.Series:
    p = os.path.join(REPO, "data", "eva", "data", "image_content_category.csv")
    d = pd.read_csv(p)
    d["image_id"] = d["image_id"].astype(str)
    return d.drop_duplicates("image_id").set_index("image_id")["sort"].astype(str)


def analyze(dataset: str, run: str, cats: pd.Series) -> dict:
    r = load_run(run)
    dim = PRIMARY_DIM[dataset]
    g = f"{dim}_gt_norm"
    df = dedup(r.df).dropna(subset=[g, f"{dim}_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    df["imageName"] = df["imageName"].astype(str)
    df["cat"] = df["imageName"].map(cats)
    per_img = df.groupby(["cat", "imageName"]).agg(pm=("cal", "mean"), om=(g, "mean"))
    rows = {}
    for cat, sub in per_img.groupby("cat"):
        if len(sub) < 20:
            continue
        mae = float((sub["pm"] - sub["om"]).abs().mean())
        rank = float(stats.spearmanr(sub["pm"], sub["om"]).statistic)
        rows[str(cat)] = {"n_images": int(len(sub)), "group_mae": round(mae, 4),
                          "group_rank": round(rank, 3)}
    ranked = dict(sorted(rows.items(), key=lambda kv: kv[1]["group_mae"]))
    return {"dataset": dataset, "primary_dim": dim, "by_category": ranked}


def main() -> None:
    report = {"PARA": analyze("PARA", "para_full", para_categories()),
              "EVA": analyze("EVA", "eva_full", eva_categories())}
    write_json(report, "content_category.json")
    print("\nWrote results/content_category.json\n")
    for ds, m in report.items():
        print(f"{ds} — calibrated group MAE by content category (best→worst):")
        for cat, v in m["by_category"].items():
            print(f"  {cat:14} n={v['n_images']:>4}  MAE={v['group_mae']:.4f}  rank={v['group_rank']:+.3f}")
        print()


if __name__ == "__main__":
    main()
