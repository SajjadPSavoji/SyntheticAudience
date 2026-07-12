"""#6 Warm-start collaborative-filtering baseline for C2 (ground-truth only, no VLM).

The plan wants a warm-start individual reference: a model that has SEEN the user. We fit the
classic additive bias model  r_hat(u,i) = mu + b_u + b_i  (the workhorse CF baseline) on 80% of
(user,image) ratings and evaluate on the held-out 20% where both the user and the image were seen
in train. This is the individual lower-bound the VLM's cold-start prediction is read against.

Normalized [0,1] scale, primary dim, seed 0. Uses the ratings already in the runs (same task set
as the VLM), so numbers are directly comparable to the VLM individual MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibration import dedup
from common import PRIMARY_DIM, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
RNG = np.random.default_rng(0)


def fit_bias(train: pd.DataFrame, y: str, n_iter: int = 15, reg: float = 5.0) -> tuple:
    mu = train[y].mean()
    b_u = {u: 0.0 for u in train["userId"].unique()}
    b_i = {i: 0.0 for i in train["imageName"].unique()}
    for _ in range(n_iter):
        # update item bias
        t = train.assign(resid=train[y] - mu - train["userId"].map(b_u))
        gi = t.groupby("imageName")["resid"].agg(["sum", "count"])
        b_i = (gi["sum"] / (reg + gi["count"])).to_dict()
        # update user bias
        t = train.assign(resid=train[y] - mu - train["imageName"].map(b_i))
        gu = t.groupby("userId")["resid"].agg(["sum", "count"])
        b_u = (gu["sum"] / (reg + gu["count"])).to_dict()
    return mu, b_u, b_i


def analyze(dataset: str, run: str) -> dict:
    df = dedup(load_run(run).df)
    dim = PRIMARY_DIM[dataset]
    y = f"{dim}_gt_norm"
    d = df[["userId", "imageName", y]].dropna().copy()
    d["userId"] = d["userId"].astype(str)

    mask = RNG.random(len(d)) < 0.8
    train, test = d[mask], d[~mask]
    seen_u, seen_i = set(train["userId"]), set(train["imageName"])
    test = test[test["userId"].isin(seen_u) & test["imageName"].isin(seen_i)]

    mu, b_u, b_i = fit_bias(train, y)
    pred = mu + test["userId"].map(b_u).fillna(0) + test["imageName"].map(b_i).fillna(0)
    pred = pred.clip(0, 1)
    mae = float((pred - test[y]).abs().mean())
    sp = float(stats.spearmanr(pred, test[y]).statistic)
    # item-only (no personalization) reference on the same test set
    item_only = (mu + test["imageName"].map(b_i).fillna(0)).clip(0, 1)
    mae_item = float((item_only - test[y]).abs().mean())
    return {"dataset": dataset, "primary_dim": dim, "n_test": int(len(test)),
            "cf_warmstart_mae": round(mae, 4), "cf_warmstart_spearman": round(sp, 4),
            "item_only_mae": round(mae_item, 4),
            "personalization_gain_mae": round(mae_item - mae, 4)}


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    write_json(report, "cf_baseline.json")
    print("\nWrote results/cf_baseline.json\n")
    hdr = f"{'dataset':8}{'n_test':>9}{'CF warm MAE':>13}{'CF spear':>10}{'item-only MAE':>15}{'persoGain':>11}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        print(f"{ds:8}{m['n_test']:>9}{m['cf_warmstart_mae']:>13.4f}"
              f"{m['cf_warmstart_spearman']:>10.3f}{m['item_only_mae']:>15.4f}"
              f"{m['personalization_gain_mae']:>11.4f}")
    print("\nCF warm-start = mu + user_bias + item_bias (has seen the user). item-only drops the")
    print("user term. persoGain = how much knowing the user helps even a classical model (small,")
    print("consistent with Exp 0: individual taste is mostly idiosyncratic).")


if __name__ == "__main__":
    main()
