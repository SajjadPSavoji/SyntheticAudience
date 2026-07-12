"""#7 Holm-Bonferroni correction across the C1 slice sweep (plan sec.9).

The plan requires a family-wise correction across the subgroup/slice tests. We take the
per-(dataset, attribute) C1 between-group separations from results/c1_separation.json, derive a
two-sided p-value for each (t-test on the Pearson r over its cells), and apply Holm at alpha=0.05.

Caveat: the gap cells are clustered by image, so the analytic p is anti-conservative — the
1000x image-clustered bootstrap CI in c1_separation.json is the primary inference; this Holm pass
is the supplementary family-wise check the plan asks for.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from common import OUT_DIR, write_json

ALPHA = 0.05


def _pval(r: float, n: int) -> float:
    if r is None or n < 3 or abs(r) >= 1:
        return 1.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * stats.t.sf(abs(t), n - 2))


def main() -> None:
    c1 = json.load(open(os.path.join(OUT_DIR, "c1_separation.json")))
    tests = []
    for ds, m in c1.items():
        if ds.startswith("_"):
            continue
        for attr, a in m["by_attribute"].items():
            fs = a["full_separation"]
            tests.append({"dataset": ds, "attribute": attr, "r": fs["corr"],
                          "n_cells": fs["n_cells"], "p": _pval(fs["corr"], fs["n_cells"])})

    # Holm step-down
    tests.sort(key=lambda t: t["p"])
    K = len(tests)
    prev = 0.0
    for i, t in enumerate(tests):
        thresh = ALPHA / (K - i)
        adj = t["p"] * (K - i)
        prev = max(prev, adj)
        t["holm_adjusted_p"] = round(min(prev, 1.0), 5)
        t["survives_holm_0.05"] = t["holm_adjusted_p"] < ALPHA
        t["p"] = round(t["p"], 6)

    report = {"alpha": ALPHA, "n_tests": K, "tests": tests}
    write_json(report, "holm.json")
    print("\nWrote results/holm.json\n")
    hdr = f"{'dataset':7}{'attribute':22}{'r':>8}{'p':>11}{'Holm p':>10}{'survives':>10}"
    print(hdr); print("-" * len(hdr))
    for t in tests:
        print(f"{t['dataset']:7}{t['attribute']:22}{t['r']:>8.3f}{t['p']:>11.2e}"
              f"{t['holm_adjusted_p']:>10.4f}{str(t['survives_holm_0.05']):>10}")
    n_ok = sum(t["survives_holm_0.05"] for t in tests)
    print(f"\n{n_ok}/{K} slice separations survive Holm at 0.05 (primary inference = bootstrap CIs).")


if __name__ == "__main__":
    main()
