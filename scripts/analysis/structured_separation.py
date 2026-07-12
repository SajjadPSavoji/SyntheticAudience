"""#5 C1 between-group separation on the STRUCTURED signals (not just the headline axis).

Extends the C1 separation test to PARA's personal signals (contentPreference, willingnessToShare)
and EVA difficulty: do groups diverge on these the way the headline score does? Reuses the C1
machinery (image-controlled slice-gap correlation, calibrated, full vs blind). Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attrs import load_attributes
from c1_separation import _gap_cells, _separation
from calibration import cross_fit_calibrate, dedup
from common import load_run

# (dataset, structured dim, slicing attribute)
TESTS = [("PARA", "contentPreference", "artExperience"),
         ("PARA", "willingnessToShare", "artExperience"),
         ("EVA", "difficulty", "photographic_level")]


def _prep(run_name: str, dataset: str, dim: str) -> pd.DataFrame:
    r = load_run(run_name)
    g = f"{dim}_gt_norm"
    df = dedup(r.df).dropna(subset=[g, f"{dim}_pred_norm"]).copy()
    df["cal"] = cross_fit_calibrate(df, dim)
    df = df.dropna(subset=["cal"])
    df["gt"] = df[g]
    df["userId"] = df["userId"].astype(str)
    return df.join(load_attributes(dataset), on="userId")


def main() -> None:
    report = []
    for ds, dim, attr in TESTS:
        full = _prep(f"{ds.lower()}_full", ds, dim)
        blind = _prep(f"{ds.lower()}_blind", ds, dim)
        fs = _separation(_gap_cells(full, attr))
        bs = _separation(_gap_cells(blind, attr))
        report.append({"dataset": ds, "dim": dim, "slice_attr": attr,
                       "full_separation": fs, "blind_separation": bs})
    from common import write_json
    write_json(report, "structured_separation.json")
    print("\nWrote results/structured_separation.json\n")
    hdr = f"{'dataset':7}{'structured dim':22}{'slice':20}{'full sep (CI)':>26}{'blind':>8}"
    print(hdr); print("-" * len(hdr))
    for m in report:
        fs, bs = m["full_separation"], m["blind_separation"]
        cistr = f"{fs['corr']} {fs['ci95']}" if fs["corr"] is not None else "n/a"
        print(f"{m['dataset']:7}{m['dim']:22}{m['slice_attr']:20}{cistr:>26}"
              f"{(bs['corr'] if bs['corr'] is not None else 0):>8.3f}")
    print("\nDoes the persona reproduce group differences on the personal/structured signals too?")


if __name__ == "__main__":
    main()
