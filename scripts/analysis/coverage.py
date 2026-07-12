"""Coverage & traceability — which source rows do the results actually cover?

The runs were not on the full datasets (they take a long time), so before quoting any
result we establish exactly what slice of each source dataset was scored, and verify that
every result record traces back to a real (image, user, ground-truth) row in the source
annotation files under data/{para,lapis,eva}.

Key per dataset (result column -> source column):
  PARA : (imageName, userId)              -> PARA-Images.csv (sessionId, imageName, userId, ...)
  EVA  : (imageName, userId)              -> votes_filtered.csv (image_id, user_id, ...)   [= delimited]
  LAPIS: (imageName, userId)              -> LAPIS_PIAA.csv (image_filename, participant_id, ...)

Writes results/coverage.json. Pure re-analysis, no inference.
"""
from __future__ import annotations

import os

import pandas as pd

from common import REPO, load_run, write_json

DATA = os.path.join(REPO, "data")


def _src_para() -> pd.DataFrame:
    p = os.path.join(DATA, "para", "annotation", "PARA-Images.csv")
    d = pd.read_csv(p)
    return d.rename(columns={"userId": "userId", "imageName": "imageName"})


def _src_eva() -> pd.DataFrame:
    p = os.path.join(DATA, "eva", "data", "votes_filtered.csv")
    d = pd.read_csv(p, sep="=")
    return d.rename(columns={"image_id": "imageName", "user_id": "userId"})


def _src_lapis() -> pd.DataFrame:
    p = os.path.join(DATA, "lapis", "annotation", "LAPIS_PIAA.csv")
    d = pd.read_csv(p)
    return d.rename(columns={"image_filename": "imageName", "participant_id": "userId"})


SRC = {"PARA": _src_para, "EVA": _src_eva, "LAPIS": _src_lapis}
GT_CHECK = {"PARA": ("aestheticScore", "gt_aestheticScore"),
            "EVA": ("score", "gt_score"),
            "LAPIS": ("rating", "gt_rating")}


def coverage_for(dataset: str, run_name: str) -> dict:
    run = load_run(run_name)
    src = SRC[dataset]()
    rdf = run.df.copy()
    # normalize key dtypes to string for robust set ops
    for df in (rdf, src):
        df["imageName"] = df["imageName"].astype(str)
        df["userId"] = df["userId"].astype(str)

    res_imgs = set(rdf["imageName"].unique())
    res_users = set(rdf["userId"].unique())
    src_imgs = set(src["imageName"].unique())
    src_users = set(src["userId"].unique())

    # ratings coverage (dedup exact rows in results first)
    cols = [c for c in rdf.columns if not c.endswith("_norm")]
    n_res_ratings = len(rdf.drop_duplicates(subset=cols))

    # traceability: do result (image,user) keys exist in source?
    res_keys = set(map(tuple, rdf[["imageName", "userId"]].itertuples(index=False, name=None)))
    src_keys = set(map(tuple, src[["imageName", "userId"]].itertuples(index=False, name=None)))
    keys_in_src = len(res_keys & src_keys)

    # gt value spot-check: merge and compare on the primary dim
    scol, gcol = GT_CHECK[dataset]
    merged = rdf.merge(src[["imageName", "userId", scol]].drop_duplicates(["imageName", "userId"]),
                       on=["imageName", "userId"], how="left", suffixes=("", "_src"))
    have = merged[gcol].notna() & merged[scol].notna()
    # LAPIS has repeated measures -> a key can map to several source ratings; allow tolerance
    gt_match = float((abs(merged.loc[have, gcol] - merged.loc[have, scol]) <= 0.5).mean())

    return {
        "dataset": dataset, "run": run_name,
        "sampling": {"scheme": run.config.get("sampling"),
                     "seed": run.config.get("seed"),
                     "raters_per_image_cap": run.config.get("raters_per_image")},
        "images": {"covered": len(res_imgs), "source_total": len(src_imgs),
                   "fraction": round(len(res_imgs) / len(src_imgs), 4),
                   "all_covered_in_source": res_imgs.issubset(src_imgs)},
        "users": {"covered": len(res_users), "source_total": len(src_users),
                  "fraction": round(len(res_users) / len(src_users), 4)},
        "ratings": {"covered": int(n_res_ratings), "source_total": int(len(src)),
                    "fraction": round(n_res_ratings / len(src), 4)},
        "traceability": {
            "result_keys": len(res_keys),
            "keys_found_in_source": keys_in_src,
            "keys_found_frac": round(keys_in_src / len(res_keys), 4),
            "gt_value_match_frac_within_0.5": round(gt_match, 4)},
    }


def main() -> None:
    report = {ds: coverage_for(ds, f"{ds.lower()}_full") for ds in SRC}
    path = write_json(report, "coverage.json")
    print(f"\nWrote {path}\n")
    hdr = (f"{'dataset':7}{'imgs cov/total':>18}{'frac':>7}"
           f"{'ratings cov/total':>22}{'frac':>7}{'keys→src':>9}{'gt✓':>7}")
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        i, r, t = m["images"], m["ratings"], m["traceability"]
        print(f"{ds:7}{i['covered']:>8}/{i['source_total']:<9}{i['fraction']:>7.3f}"
              f"{r['covered']:>10}/{r['source_total']:<11}{r['fraction']:>7.3f}"
              f"{t['keys_found_frac']:>9.3f}{t['gt_value_match_frac_within_0.5']:>7.2f}")
    print("\nimgs/ratings frac = fraction of the SOURCE dataset the runs actually scored.")
    print("keys→src = fraction of result (image,user) rows found in the source annotation file.")
    print("gt✓ = fraction whose stored ground truth matches the source rating (±0.5).")


if __name__ == "__main__":
    main()
