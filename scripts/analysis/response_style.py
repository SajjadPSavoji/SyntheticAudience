"""#2 VLM response-style audit — how does the judge use the rating scale?

We have characterized the mean bias but never the SHAPE of the judge's output. Does it use the
full scale or pile on the middle? Avoid the endpoints? For the primary dim (native grid) we
compare the VLM's marginal predicted distribution to the human one:
  - std (scale spread)      - Shannon entropy over grid values (how much of the scale is used)
  - modal share (peakiness) - endpoint share (fraction at the extreme low/high grid values)
Explains mechanistically why calibration helps and why raw EMD is large. Pure re-analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calibration import dedup
from common import PRIMARY_DIM, SCALES, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full", "LAPIS": "lapis_full"}


def _entropy(vals: np.ndarray, bins: np.ndarray) -> float:
    h, _ = np.histogram(vals, bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _describe(vals: np.ndarray, lo: float, hi: float) -> dict:
    n = len(vals)
    # grid edges: treat integer/half grid; use unit bins across [lo,hi]
    edges = np.arange(lo - 0.5, hi + 1.0, 1.0) if (hi - lo) <= 20 else np.linspace(lo, hi, 21)
    vc = pd.Series(vals).value_counts(normalize=True)
    endpoint = float(((vals <= lo + 1e-9) | (vals >= hi - 1e-9)).mean())
    return {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals)), 3),
            "entropy_bits": round(_entropy(vals, edges), 3),
            "modal_share": round(float(vc.max()), 3),
            "endpoint_share": round(endpoint, 3)}


def analyze(dataset: str, run: str) -> dict:
    df = dedup(load_run(run).df)
    dim = PRIMARY_DIM[dataset]
    lo, hi = SCALES[dataset][dim]
    d = df.dropna(subset=[f"gt_{dim}", f"pred_{dim}"])
    gt = _describe(d[f"gt_{dim}"].to_numpy(), lo, hi)
    pred = _describe(d[f"pred_{dim}"].to_numpy(), lo, hi)
    return {"dataset": dataset, "dim": dim, "scale": [lo, hi],
            "human": gt, "vlm": pred,
            "scale_usage_ratio": round(pred["std"] / gt["std"], 3),
            "entropy_ratio": round(pred["entropy_bits"] / gt["entropy_bits"], 3)}


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    write_json(report, "response_style.json")
    print("\nWrote results/response_style.json\n")
    hdr = (f"{'dataset':7}{'who':>6}{'mean':>7}{'std':>7}{'entropy':>9}"
           f"{'modal%':>8}{'endpt%':>8}")
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        for who in ("human", "vlm"):
            s = m[who]
            print(f"{ds if who=='human' else '':7}{who:>6}{s['mean']:>7.2f}{s['std']:>7.3f}"
                  f"{s['entropy_bits']:>9.2f}{s['modal_share']:>8.2f}{s['endpoint_share']:>8.2f}")
        print(f"{'':7}{'ratio':>6}{'':7}{m['scale_usage_ratio']:>7.2f}{m['entropy_ratio']:>9.2f}")
        print("-" * len(hdr))
    print("scale_usage/entropy ratio < 1 => VLM compresses toward the middle (central tendency),")
    print("under-uses the scale and endpoints -> raw EMD inflated, calibration recovers it.")


if __name__ == "__main__":
    main()
