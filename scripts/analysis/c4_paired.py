"""C4 paired comparison — AutoPolish against every other critic, image by image (Table 3).

All four critics edit the same 100 source images, so each comparison is paired. For each
image and each other critic we take d = metric(society) - metric(other) and report:

  1. the mean of d with a 95% paired bootstrap CI. Images are resampled with replacement,
     separately within PARA and within EVA (50 each), and the same draw serves both
     critics, which is what makes it paired;
  2. a paired p-value: Wilcoxon signed-rank for the final gain and the edit size, and the
     exact McNemar test for the share of improved images, whose paired differences are
     only -1, 0 or +1 (Wilcoxon on those reduces to a sign test, and McNemar is its exact
     form);
  3. Holm over the three comparisons (static, blind, reward-only) within each metric.

Gain per improved image is not tested: each critic improves a different subset of images,
so the two means are not paired. It stays descriptive.

The run has one seed, so the bootstrap covers variation over images, not over seeds.

Pure re-analysis: no GPU/inference. Run from ``scripts/analysis/``.

    python c4_paired.py                         # reads data/results/c4_run2/logs
    python c4_paired.py --logs-dir <dir>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats

from c4_trajectory import _finals, load_c4
from common import REPO, write_json

DEFAULT_LOGS = os.path.join(REPO, "data", "results", "c4_run2", "logs")
OURS = "society"
OTHERS = ["static", "blind", "reward_only"]
N_BOOT = 10_000
ALPHA = 0.05


def _per_image(finals: dict) -> dict[str, dict[str, float]]:
    """image_id -> the three paired metrics of one critic."""
    return {i: {"gain": f["gain"], "improved": float(f["gain"] > 0),
                "edit_size": 1.0 - f["drift_final"]} for i, f in finals.items()}


def _paired_ci(d: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% CI of mean(d), resampling images with replacement within each stratum."""
    draws = np.zeros(N_BOOT)
    for s in np.unique(strata):
        ds = d[strata == s]
        draws += ds[rng.integers(0, len(ds), size=(N_BOOT, len(ds)))].sum(axis=1)
    draws /= len(d)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p for paired binary outcomes a, b."""
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def _holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (same rule as holm.py)."""
    order = np.argsort(ps)
    adj, prev, K = [0.0] * len(ps), 0.0, len(ps)
    for i, j in enumerate(order):
        prev = max(prev, ps[j] * (K - i))
        adj[j] = min(prev, 1.0)
    return adj


def analyze(logs_dir: str) -> dict:
    per = {c: _per_image(_finals(load_c4(c, logs_dir))) for c in [OURS] + OTHERS}
    ids = sorted(per[OURS])
    for c in OTHERS:
        if sorted(per[c]) != ids:
            raise SystemExit(f"{c} and {OURS} were not run on the same images.")
    strata = np.array([i.split("__")[0] for i in ids])
    rng = np.random.default_rng(0)

    res: dict = {"n_images": len(ids), "n_boot": N_BOOT,
                 "strata": {str(s): int((strata == s).sum()) for s in np.unique(strata)},
                 "difference": f"{OURS} minus other", "metrics": {}}
    for m in ("gain", "improved", "edit_size"):
        a = np.array([per[OURS][i][m] for i in ids])
        rows = {}
        for c in OTHERS:
            b = np.array([per[c][i][m] for i in ids])
            d = a - b
            if m == "improved":
                p, test = _mcnemar_exact(a, b), "mcnemar_exact"
            else:
                p, test = float(stats.wilcoxon(a, b).pvalue), "wilcoxon_signed_rank"
            rows[c] = {"ours": float(a.mean()), "other": float(b.mean()),
                       "mean_diff": float(d.mean()), "ci95": _paired_ci(d, strata, rng),
                       "n_ours_higher": int((d > 0).sum()), "n_other_higher": int((d < 0).sum()),
                       "n_tied": int((d == 0).sum()), "test": test, "p": p}
        for c, ph in zip(OTHERS, _holm([rows[c]["p"] for c in OTHERS])):
            rows[c]["holm_p"] = ph
            rows[c]["ci_excludes_zero"] = not (rows[c]["ci95"][0] <= 0 <= rows[c]["ci95"][1])
        res["metrics"][m] = rows
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs-dir", default=DEFAULT_LOGS)
    args = p.parse_args()
    res = analyze(args.logs_dir)
    print(f"{OURS} minus each critic, {res['n_images']} images {res['strata']}, "
          f"{N_BOOT} paired resamples within dataset")
    hdr = f"{'metric':10}{'vs':13}{'diff':>9}{'95% CI':>22}{'+/-/=':>12}{'p':>10}{'Holm p':>9}"
    print(hdr); print("-" * len(hdr))
    for m, rows in res["metrics"].items():
        for c, r in rows.items():
            ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
            cnt = f"{r['n_ours_higher']}/{r['n_other_higher']}/{r['n_tied']}"
            print(f"{m:10}{c:13}{r['mean_diff']:>+9.3f}{ci:>22}{cnt:>12}"
                  f"{r['p']:>10.4f}{r['holm_p']:>9.4f}")
    print("wrote", write_json(res, "c4_paired.json"))


if __name__ == "__main__":
    main()
