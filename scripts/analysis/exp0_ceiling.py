"""Exp 0 — ceiling analysis (research_plan.md sec.4). Pure re-analysis of the human gt.

The question Exp 0 answers: how predictable is individual taste *at all*, and how much
more predictable is the group mean? We decompose the human rating variance into a
between-image (shared, predictable) part and a within-image (idiosyncratic) part with a
one-way random-effects model (raters nested in images, unequal group sizes), then contrast
the reliability of a *single* rating against the reliability of the *k-rater group mean*.

Reported per dataset on the normalized [0,1] scale so datasets are comparable:
  - ICC(1)      = reliability of ONE rating          = between-image variance fraction
  - ICC(k)      = reliability of the k-rater MEAN     (Spearman-Brown up-projection)
  - the individual-vs-group reliability gap          (the motivation for aggregation)
  - noise floors: mean per-image human std vs the std error of the image mean

Persona-attributable variance (ΔR^2) needs the userId->attribute join and is deferred to
the steerability / C1 workstream (B1/B2); we note it but do not fake it here.

Ground truth is identical across full/blind, so we read it once from the *_full run.
Exact-duplicate rows (the LAPIS export artifact found in the audit) are dropped first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from attrs import ATTRS, load_attributes
from common import PRIMARY_DIM, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-identical duplicate rows (export artifact); keep genuine repeats."""
    cols = [c for c in df.columns if not c.endswith("_norm")]
    return df.drop_duplicates(subset=cols)


def oneway_variance(df: pd.DataFrame, val: str, group: str) -> dict:
    """One-way random-effects decomposition of `val` grouped by `group`.

    Handles unequal group sizes. Returns variance components + ICC(1) and ICC(k).
    """
    g = df.groupby(group)[val]
    n_i = g.size().to_numpy(dtype=float)
    mean_i = g.mean().to_numpy()
    N = n_i.sum()
    k = len(n_i)
    grand = df[val].mean()

    ss_between = float((n_i * (mean_i - grand) ** 2).sum())
    # within: total SS - between SS
    ss_total = float(((df[val] - grand) ** 2).sum())
    ss_within = ss_total - ss_between
    df_b, df_w = k - 1, N - k
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w if df_w > 0 else np.nan

    # average group size correction for unequal n
    k0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    var_between = max((ms_b - ms_w) / k0, 0.0)
    var_within = ms_w
    icc1 = (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)      # reliability of 1 rating
    icc_k = (ms_b - ms_w) / ms_b                          # reliability of the k-mean
    return {
        "n_groups": int(k), "n_obs": int(N), "k0_mean_group_size": round(k0, 2),
        "var_between": var_between, "var_within": var_within,
        "var_total": var_between + var_within,
        "between_fraction": var_between / (var_between + var_within),
        "within_fraction": var_within / (var_between + var_within),
        "ICC1_single_rating": float(icc1),
        "ICCk_group_mean": float(icc_k),
    }


def persona_delta_r2(dataset: str, df: pd.DataFrame, col: str) -> dict:
    """How much of the WITHIN-image (idiosyncratic) rating variance do the rater's
    persona attributes explain? Image-center the gt (remove the shared image effect),
    then regress the residual on one-hot/numeric persona attributes; R^2 is the fraction
    of within-image variance that persona explains (the persona-attributable ceiling)."""
    attrs = load_attributes(dataset)
    d = df.copy()
    d["userId"] = d["userId"].astype(str)
    d = d.join(attrs, on="userId")
    d["resid"] = d[col] - d.groupby("imageName")[col].transform("mean")

    # build the persona design matrix: numeric cols as-is, categoricals one-hot
    feats = []
    for a in ATTRS[dataset]:
        if a not in d.columns:
            continue
        num = pd.to_numeric(d[a], errors="coerce")
        if num.notna().mean() > 0.9 and num.nunique() > 6:
            feats.append(num.fillna(num.median()).rename(a))
        else:
            feats.append(pd.get_dummies(d[a].astype(str), prefix=a, dummy_na=True))
    X = pd.concat(feats, axis=1).astype(float)
    y = d["resid"].to_numpy()
    within_var = float(np.var(y))
    total_var = float(np.var(d[col].to_numpy()))
    reg = LinearRegression().fit(X.to_numpy(), y)
    r2_within = float(reg.score(X.to_numpy(), y))   # frac of within-image var explained
    # as a fraction of TOTAL variance
    delta_r2_total = r2_within * (within_var / total_var)
    return {
        "persona_delta_R2_within_image": round(r2_within, 4),
        "persona_delta_R2_total": round(delta_r2_total, 4),
        "n_features": int(X.shape[1]),
        "note": "in-sample OLS R^2 -> an upper estimate of the persona-attributable ceiling.",
    }


def ceiling_for(dataset: str, run_name: str) -> dict:
    r = load_run(run_name)
    dim = PRIMARY_DIM[dataset]
    col = f"{dim}_gt_norm"
    df = dedup(r.df)[["imageName", "userId", col]].dropna(subset=[col]).copy()

    dec = oneway_variance(df, col, "imageName")

    # noise-floor contrast: per-image human std vs std error of the image mean
    per_img = df.groupby("imageName")[col]
    std_i = per_img.std(ddof=1)
    n_i = per_img.size()
    sem_i = std_i / np.sqrt(n_i)
    mean_human_std = float(std_i.mean())
    mean_sem = float(sem_i.mean())

    n_users = df["userId"].nunique()
    persona = persona_delta_r2(dataset, df, col)
    return {
        "dataset": dataset, "primary_dim": dim, "scale": "normalized [0,1]",
        "n_ratings": int(len(df)), "n_images": int(df["imageName"].nunique()),
        "n_users": int(n_users),
        "raters_per_image": {
            "mean": round(float(n_i.mean()), 2),
            "median": float(n_i.median()),
            "min": int(n_i.min()), "max": int(n_i.max())},
        "variance_decomposition": dec,
        "persona_variance": persona,
        "noise_floor": {
            "mean_within_image_human_std": round(mean_human_std, 4),
            "mean_std_error_of_image_mean": round(mean_sem, 4),
            "individual_vs_group_noise_ratio": round(mean_human_std / mean_sem, 2)},
        "interpretation": {
            "individual_ceiling_R2_from_image_only": round(dec["between_fraction"], 4),
            "single_rating_reliability_ICC1": round(dec["ICC1_single_rating"], 4),
            "group_mean_reliability_ICCk": round(dec["ICCk_group_mean"], 4),
            "reliability_gain_from_aggregation":
                round(dec["ICCk_group_mean"] - dec["ICC1_single_rating"], 4),
            "note": "persona-attributable variance (delta-R2) requires the userId->"
                    "attribute join; deferred to B1/B2 (steerability / C1)."},
    }


def main() -> None:
    report = {ds: ceiling_for(ds, run) for ds, run in DATASETS.items()}
    path = write_json(report, "exp0.json")
    print(f"\nWrote {path}\n")

    hdr = (f"{'dataset':8}{'ratings':>9}{'imgs':>6}{'k0':>6}"
           f"{'betwFrac':>9}{'ICC1':>7}{'ICCk':>7}{'noiseRatio':>11}{'personaΔR²(win)':>16}")
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        d = m["variance_decomposition"]
        print(f"{ds:8}{m['n_ratings']:>9}{m['n_images']:>6}{d['k0_mean_group_size']:>6.1f}"
              f"{d['between_fraction']:>9.3f}{d['ICC1_single_rating']:>7.3f}"
              f"{d['ICCk_group_mean']:>7.3f}"
              f"{m['noise_floor']['individual_vs_group_noise_ratio']:>11.1f}"
              f"{m['persona_variance']['persona_delta_R2_within_image']:>16.4f}")
    print("\nReading: between_fraction / ICC1 = how much of an INDIVIDUAL rating is shared,")
    print("predictable signal (the rest is idiosyncratic). ICCk = reliability of the GROUP")
    print("mean. The ICC1->ICCk jump is exactly why the group is predictable where the")
    print("individual is not (C2). noiseRatio = individual noise / group-mean noise.")


if __name__ == "__main__":
    main()
