"""#5 Inter-dimension correlation structure. Pure re-analysis.

Human aesthetic judgment has structure: aesthetic appeal co-varies with quality, composition,
etc. Does the frozen VLM reproduce that structure? For every pair of rated dimensions we compute
the correlation across images (per-image mean) for humans vs for the VLM, then compare the two
correlation matrices (off-diagonal): their correlation, and mean |Δ|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from calibration import dedup
from common import SCALES, load_run, write_json

DATASETS = {"PARA": "para_full", "EVA": "eva_full"}


def _corr_offdiag(mat: pd.DataFrame) -> np.ndarray:
    m = mat.to_numpy()
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def analyze(dataset: str, run: str) -> dict:
    df = dedup(load_run(run).df)
    dims = [d for d in SCALES[dataset] if f"{d}_gt_norm" in df.columns]
    per_img = df.groupby("imageName").agg(
        **{f"h_{d}": (f"{d}_gt_norm", "mean") for d in dims},
        **{f"v_{d}": (f"{d}_pred_norm", "mean") for d in dims})
    H = per_img[[f"h_{d}" for d in dims]].corr()
    V = per_img[[f"v_{d}" for d in dims]].corr()
    h_off, v_off = _corr_offdiag(H), _corr_offdiag(V)
    return {
        "dataset": dataset, "n_dims": len(dims), "dims": dims,
        "structure_match_corr": round(float(stats.pearsonr(h_off, v_off).statistic), 4),
        "mean_abs_diff_offdiag": round(float(np.mean(np.abs(h_off - v_off))), 4),
        "human_mean_offdiag_corr": round(float(np.mean(h_off)), 4),
        "vlm_mean_offdiag_corr": round(float(np.mean(v_off)), 4),
    }


def main() -> None:
    report = {ds: analyze(ds, run) for ds, run in DATASETS.items()}
    write_json(report, "structure.json")
    print("\nWrote results/structure.json\n")
    hdr = f"{'dataset':8}{'dims':>6}{'structMatch':>13}{'mean|Δ|':>10}{'humanCorr':>11}{'vlmCorr':>9}"
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        print(f"{ds:8}{m['n_dims']:>6}{m['structure_match_corr']:>13.3f}"
              f"{m['mean_abs_diff_offdiag']:>10.3f}{m['human_mean_offdiag_corr']:>11.3f}"
              f"{m['vlm_mean_offdiag_corr']:>9.3f}")
    print("\nstructMatch = corr between human and VLM inter-dimension correlation patterns.")
    print("humanCorr/vlmCorr = average cross-dimension coupling (VLM often over-couples axes).")


if __name__ == "__main__":
    main()
