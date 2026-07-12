"""#4 Leakage / memorization diagnostics (plan sec.8.5). Pure re-analysis.

LAPIS art is WikiArt — the VLM may have seen famous works/artists in pretraining. If so, it
should rate FREQUENT (famous) artists/styles more accurately. We join artist/style from
LAPIS_PIAA, compute per-artist calibrated MAE, and correlate artist rating-count with error
(a negative correlation = a memorization/familiarity signal). Also per-style MAE spread.

EVA images are an AVA subset; we note the overlap risk (our pipeline is zero-shot with no AVA
exemplars, so there is no train/test leakage path here) rather than a numeric probe.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from calibration import cross_fit_calibrate, dedup
from common import REPO, load_run, write_json


def lapis_meta() -> pd.DataFrame:
    p = os.path.join(REPO, "data", "lapis", "annotation", "LAPIS_PIAA.csv")
    d = pd.read_csv(p, usecols=["image_filename", "artist", "style", "genre"])
    d = d.rename(columns={"image_filename": "imageName"}).drop_duplicates("imageName")
    return d.set_index("imageName")


def main() -> None:
    r = load_run("lapis_full")
    df = dedup(r.df).dropna(subset=["rating_gt_norm", "rating_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, "rating")
    df = df.dropna(subset=["cal"])
    df["abserr"] = (df["cal"] - df["rating_gt_norm"]).abs()
    df = df.join(lapis_meta(), on="imageName")

    # per-artist: rating volume vs error
    art = df.groupby("artist").agg(n=("abserr", "size"), mae=("abserr", "mean"))
    art = art[art["n"] >= 20]
    corr_n_err = float(stats.spearmanr(art["n"], art["mae"]).statistic)

    sty = df.groupby("style").agg(n=("abserr", "size"), mae=("abserr", "mean"))
    sty = sty[sty["n"] >= 50].sort_values("mae")

    report = {
        "dataset": "LAPIS", "n_artists>=20": int(len(art)),
        "spearman_artist_volume_vs_error": round(corr_n_err, 4),
        "interpretation": "negative => more-rated (famous) artists are rated MORE accurately, "
                          "a familiarity/memorization signal; ~0 => no such effect.",
        "style_mae_best": {str(k): round(float(v), 4) for k, v in sty["mae"].head(3).items()},
        "style_mae_worst": {str(k): round(float(v), 4) for k, v in sty["mae"].tail(3).items()},
        "style_mae_gap": round(float(sty["mae"].max() - sty["mae"].min()), 4),
        "eva_ava_note": "EVA images are an AVA subset; pipeline is zero-shot with no AVA "
                        "exemplars, so there is no train/test leakage path (documented, not scored).",
    }
    write_json(report, "leakage.json")
    print("\nWrote results/leakage.json\n")
    print(f"LAPIS memorization probe ({report['n_artists>=20']} artists >=20 ratings):")
    print(f"  Spearman(artist rating-volume, error) = {corr_n_err:+.3f}")
    print(f"  ({report['interpretation']})")
    print(f"\n  style MAE gap (worst-best) = {report['style_mae_gap']:.4f}")
    print(f"  best styles : {report['style_mae_best']}")
    print(f"  worst styles: {report['style_mae_worst']}")


if __name__ == "__main__":
    main()
