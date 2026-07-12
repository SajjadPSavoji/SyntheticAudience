"""#9 Off-grid / snapping parse audit.

The pipeline snaps each raw model score to the dimension's value grid. We never quantified how
often the raw model output was off-grid or how often snapping changed the value. We re-parse the
primary score from raw_response and compare it to the grid and to the stored pred. Pure re-analysis.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import PRIMARY_DIM, SCALES, load_run, write_json

RUNS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}
STEP = {"aestheticScore": 0.5, "score": 1.0, "rating": 1.0}


def _raw_score(raw: str, key: str):
    if not isinstance(raw, str):
        return None
    try:
        s = raw[raw.index("{"): raw.rindex("}") + 1]
        return float(json.loads(s)[key])
    except Exception:
        m = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
        return float(m.group(1)) if m else None


def analyze(dataset: str, run: str) -> dict:
    df = load_run(run).df
    dim = PRIMARY_DIM[dataset]
    lo, hi = SCALES[dataset][dim]
    step = STEP[dim]
    raw = df["raw_response"].map(lambda r: _raw_score(r, dim))
    pred = pd.to_numeric(df[f"pred_{dim}"], errors="coerce")

    parsed = raw.notna()
    r = raw[parsed].to_numpy()
    grid = np.round((r - lo) / step) * step + lo
    on_grid = np.isclose(r, grid, atol=1e-6)
    out_of_range = (r < lo) | (r > hi)
    snapped_changed = ~np.isclose(raw[parsed].to_numpy(), pred[parsed].to_numpy(), atol=1e-6)
    return {"dataset": dataset, "dim": dim, "n": int(len(df)),
            "raw_parse_rate": round(float(parsed.mean()), 5),
            "off_grid_frac": round(float((~on_grid).mean()), 5),
            "out_of_range_frac": round(float(out_of_range.mean()), 5),
            "snapping_changed_frac": round(float(snapped_changed.mean()), 5)}


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in RUNS.items()}
    write_json(report, "offgrid_audit.json")
    print("\nWrote results/offgrid_audit.json\n")
    hdr = f"{'dataset':7}{'rawParse':>10}{'offGrid':>9}{'outRange':>10}{'snapChanged':>13}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        print(f"{ds:7}{m['raw_parse_rate']:>10.4f}{m['off_grid_frac']:>9.4f}"
              f"{m['out_of_range_frac']:>10.4f}{m['snapping_changed_frac']:>13.4f}")
    print("\nLow off-grid/out-of-range/snap-changed => the model already answers on the intended")
    print("grid and snapping is nearly a no-op (the discretization isn't distorting results).")


if __name__ == "__main__":
    main()
