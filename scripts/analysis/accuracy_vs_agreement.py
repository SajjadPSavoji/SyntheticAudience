"""#3 Accuracy vs human agreement — does the model do better where humans agree?

Links the Exp-0 ceiling to observed performance: images where raters agree (low score std)
should be predicted more accurately by the aggregate. Per image we take human disagreement
(std of gt) and the VLM's calibrated group error (|group pred mean - obs mean|), correlate them,
and show mean error across agreement terciles. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibration import cross_fit_calibrate, dedup
from common import PRIMARY_DIM, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}


def analyze(dataset: str, run: str) -> dict:
    r = load_run(run)
    dim = PRIMARY_DIM[dataset]
    g = f"{dim}_gt_norm"
    df = dedup(r.df).dropna(subset=[g, f"{dim}_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    per = df.groupby("imageName").agg(dis=(g, "std"), pm=("cal", "mean"),
                                      om=(g, "mean"), n=(g, "size")).dropna()
    per = per[per["n"] >= 10]
    per["err"] = (per["pm"] - per["om"]).abs()
    rho = float(stats.spearmanr(per["dis"], per["err"]).statistic)
    per["tercile"] = pd.qcut(per["dis"], 3, labels=["low_disagree", "mid", "high_disagree"])
    terc = per.groupby("tercile", observed=True)["err"].mean().round(4).to_dict()
    return {"dataset": dataset, "n_images": int(len(per)),
            "corr_disagreement_vs_error": round(rho, 4),
            "group_error_by_agreement_tercile": {k: round(float(v), 4) for k, v in terc.items()}}


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    write_json(report, "accuracy_vs_agreement.json")
    print("\nWrote results/accuracy_vs_agreement.json\n")
    hdr = f"{'dataset':7}{'corr(dis,err)':>15}{'err low-dis':>13}{'err mid':>10}{'err high-dis':>14}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        t = m["group_error_by_agreement_tercile"]
        print(f"{ds:7}{m['corr_disagreement_vs_error']:>15.3f}"
              f"{t.get('low_disagree',float('nan')):>13.4f}{t.get('mid',float('nan')):>10.4f}"
              f"{t.get('high_disagree',float('nan')):>14.4f}")
    print("\ncorr>0 and error rising low->high disagreement => the model is most accurate exactly")
    print("where humans agree, i.e. performance tracks the Exp-0 ceiling.")


if __name__ == "__main__":
    main()
