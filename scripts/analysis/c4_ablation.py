"""C4 ablations — replay the AutoPolish acceptance rule from the logged runs.

Every round of ``script/c4_refine.py`` logs all K candidates with their held-out
aesthetic score and DINOv2 similarity to the source, whether or not they pass the
similarity floor. That lets us re-apply a different acceptance rule offline,
without new edits:

  1. accept-if-better on / off (commit each round's best candidate regardless);
  2. candidates per round, K = 1, 2, 3 (use only the first k logged seeds);
  3. similarity floor swept from none to 0.94;
  4. identity: how far final images drift once the floor is removed.

Each round's instruction is held as logged. That is exact for the static and
reward-only critics, whose instruction never depends on the image, and an
approximation for blind and society, whose critic looks at the current best image.

Pure re-analysis: no GPU/inference. Run from ``scripts/analysis/``.

    python c4_ablation.py                       # reads data/results/c4_run2/logs
    python c4_ablation.py --logs-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from common import REPO, write_json

CONDITIONS = ["static", "blind", "society", "reward_only"]
DEFAULT_LOGS = os.path.join(REPO, "data", "results", "c4_run2", "logs")
FLOOR = 0.78
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_c4(condition: str, logs_dir: str) -> dict[str, list[dict]]:
    """image_id -> step records (step 0 = source), sorted by step."""
    run = f"c4_{condition}"
    by_img: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(logs_dir, run, f"{run}*.part-*.json"))):
        with open(p, encoding="utf-8") as f:
            for r in json.load(f):
                by_img[r["image_id"]].append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in by_img.items()}


def replay(steps: list[dict], k: int = 3, floor: float = FLOOR, gate: bool = True):
    """Final (gain, similarity) under one acceptance rule, instructions held fixed."""
    src = steps[0]["best_obj"]
    best, sim = src, 1.0
    for s in steps[1:]:
        cands = [c for c in s["candidates"][:k] if c["drift"] >= floor]
        if not cands:
            continue
        c = max(cands, key=lambda c: c["aesthetic"])
        if not gate or c["aesthetic"] > best:
            best, sim = c["aesthetic"], c["drift"]
    return best - src, sim


def _ci(x: np.ndarray) -> list[float]:
    boot = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def summarize(runs: dict[str, list[dict]], **rule) -> dict:
    out = np.array([replay(v, **rule) for v in runs.values()])
    gain, sim = out[:, 0], out[:, 1]
    imp = gain > 1e-9
    below = sim < FLOOR
    return {
        "gain_below_floor": float(gain[below].mean()) if below.any() else None,
        "gain_rest": float(gain[~below].mean()),
        "gain": float(gain.mean()), "gain_ci95": _ci(gain),
        "improved": float(imp.mean()),
        "gain_per_improved": float(gain[imp].mean()) if imp.any() else 0.0,
        "edit_size": float(1 - sim.mean()),
        "frac_final_below_floor": float((sim < FLOOR).mean()),
        "min_similarity": float(sim.min()),
        "n_images": int(len(gain)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs-dir", default=DEFAULT_LOGS)
    args = p.parse_args()

    runs = {c: load_c4(c, args.logs_dir) for c in CONDITIONS}
    res: dict = {"logs_dir": args.logs_dir, "default_floor": FLOOR, "conditions": {}}
    for c, data in runs.items():
        logged = float(np.mean([v[-1]["best_obj"] - v[0]["best_obj"] for v in data.values()]))
        sims = np.array([cd["drift"] for v in data.values() for s in v[1:] for cd in s["candidates"]])
        res["conditions"][c] = {
            "logged_gain": logged,
            "default": summarize(data),
            "gate_off": summarize(data, gate=False),
            "gate_off_k1": summarize(data, k=1, gate=False),
            "candidates": {str(k): summarize(data, k=k) for k in (1, 2, 3)},
            "floor": {f"{f:.2f}": summarize(data, floor=f) for f in (0.0, 0.70, 0.74, FLOOR, 0.82, 0.86, 0.90, 0.94)},
            "frac_candidates_below_floor": float((sims < FLOOR).mean()),
            "n_candidates": int(len(sims)),
        }
        d = res["conditions"][c]
        print(f"{c:12s} logged {logged:+.3f} replay {d['default']['gain']:+.3f} | "
              f"gate off {d['gate_off']['gain']:+.3f} | K=1 {d['candidates']['1']['gain']:+.3f} | "
              f"no floor {d['floor']['0.00']['gain']:+.3f} (edit {d['floor']['0.00']['edit_size']:.3f}, "
              f"below floor {d['floor']['0.00']['frac_final_below_floor']:.0%}) | "
              f"rejected {d['frac_candidates_below_floor']:.1%}")
    print("wrote", write_json(res, "c4_ablation.json"))


if __name__ == "__main__":
    main()
