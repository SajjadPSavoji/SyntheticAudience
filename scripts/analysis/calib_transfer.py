"""#6 Calibration cross-dataset transfer — does 'fit once, reuse' hold (needed for C3)?

The plan fits calibration on real images and reuses it unchanged for generated images (C3). As a
proxy for that transfer, we fit an isotonic map on dataset A (all data, normalized [0,1]) and apply
it to dataset B, measuring B's calibrated group MAE. The diagonal (fit on self) is the best case;
off-diagonal shows how much is lost when the calibrator was trained elsewhere. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from calibration import dedup
from common import PRIMARY_DIM, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}


def _load(ds: str) -> pd.DataFrame:
    dim = PRIMARY_DIM[ds]
    df = dedup(load_run(DATASETS[ds]).df).dropna(subset=[f"{dim}_gt_norm", f"{dim}_pred_norm"])
    return pd.DataFrame({"pred": df[f"{dim}_pred_norm"].to_numpy(),
                         "gt": df[f"{dim}_gt_norm"].to_numpy(),
                         "img": df["imageName"].to_numpy()})


def _group_mae(d: pd.DataFrame, col: str) -> float:
    g = d.groupby("img").agg(pm=(col, "mean"), om=("gt", "mean"))
    return float((g["pm"] - g["om"]).abs().mean())


def main() -> None:
    data = {ds: _load(ds) for ds in DATASETS}
    isos = {}
    for ds, d in data.items():
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(d["pred"].to_numpy(), d["gt"].to_numpy())
        isos[ds] = iso

    matrix, raw = {}, {}
    for b in DATASETS:                       # evaluated dataset
        d = data[b].copy()
        raw[b] = round(_group_mae(d, "pred"), 4)
        matrix[b] = {}
        for a in DATASETS:                   # calibrator source
            d["capp"] = isos[a].predict(d["pred"].to_numpy())
            matrix[b][a] = round(_group_mae(d, "capp"), 4)

    report = {"raw_group_mae": raw, "calibrated_group_mae_[eval][fitOn]": matrix}
    write_json(report, "calib_transfer.json")
    print("\nWrote results/calib_transfer.json\n")
    hdr = f"{'eval\\fitOn':12}" + "".join(f"{a:>9}" for a in DATASETS) + f"{'raw':>9}"
    print(hdr); print("-" * len(hdr))
    for b in DATASETS:
        row = "".join(f"{matrix[b][a]:>9.4f}" for a in DATASETS)
        print(f"{b:12}{row}{raw[b]:>9.4f}")
    print("\nRows=dataset being scored, cols=dataset the calibrator was fit on; last col=uncalibrated.")
    print("Off-diagonal close to the diagonal => calibration transfers (supports fit-once for C3).")


if __name__ == "__main__":
    main()
