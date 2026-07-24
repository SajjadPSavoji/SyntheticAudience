"""C4 self-refine — RE-SCORE the saved images with an INDEPENDENT objective.

The teammate's run (data/results/c4_selfrefine/) logs only the persona's own
aesthetic score (proposer == evaluator → circular, and it comes out flat /
regression-to-the-mean). But the edited images themselves are saved. This script
re-scores every iteration image with models the loop never used:

  - LAION aesthetic predictor (CLIP ViT-L/14 head)  -> a NEUTRAL quality yardstick
  - DINOv2 CLS cosine vs the initial image          -> how far identity has drifted

so we get a *non-circular* aesthetic trajectory + a drift trajectory over the 10
compounding edits, and can check whether the persona's self-score tracked the
neutral metric at all. This is a single-condition result (no society/blind), but
it is real — and it doubles as the "naive baseline" for the proper C4 loop.

Needs the GPU stack (torch/transformers). Run on the node, or locally with
`--device cpu` (slow but works). Reads data/results/c4_selfrefine/, writes
results/c4_selfrefine_rescore.json + figures.

    python c4_selfrefine_rescore.py               # cuda
    python c4_selfrefine_rescore.py --device cpu  # local, no GPU
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, REPO, ensure_out, write_json

sys.path.insert(0, os.path.join(REPO, "src"))
RUN = os.path.join(REPO, "data", "results", "c4_selfrefine")
RNG = np.random.default_rng(0)
N_BOOT = 1000


def _iter_files(folder: str) -> list[str]:
    """Ordered iter_00_initial.jpg, iter_01.jpg … present in a folder."""
    files = {}
    for p in glob.glob(os.path.join(folder, "iter_*.jpg")):
        b = os.path.basename(p)
        idx = 0 if "initial" in b else int(b.split("_")[1].split(".")[0])
        files[idx] = p
    return [files[i] for i in sorted(files)]


def _boot_ci(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    d = np.nanmean(x[idx], axis=1)
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score c4_selfrefine images with an independent objective.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=None, help="cap folders (debug).")
    args = ap.parse_args()
    ensure_out()

    from editor import AestheticObjective, DriftMetric
    dtype = None if args.device.startswith("cuda") else __import__("torch").float32
    print(f"loading aesthetic + drift on {args.device} …")
    obj = AestheticObjective(device=args.device, dtype=dtype)
    drift = DriftMetric(device=args.device, dtype=dtype)

    folders = sorted(glob.glob(os.path.join(RUN, "session*")))
    if args.max_images:
        folders = folders[: args.max_images]
    self_scores = {im["folder"]: im["score_trajectory"]
                   for im in json.load(open(os.path.join(RUN, "summary.json")))["images"]}

    aes_rows, drift_rows, self_rows = [], [], []
    for k, folder in enumerate(folders, 1):
        files = _iter_files(folder)
        if len(files) < 2:
            continue
        aes = obj.score_batch(files)                       # independent aesthetic per iter
        dr = [1.0] + drift.similarity_batch(files[0], files[1:])  # identity vs initial
        aes_rows.append(aes)
        drift_rows.append(dr)
        self_rows.append(self_scores.get(os.path.basename(folder), [np.nan] * len(files)))
        if k % 10 == 0:
            print(f"  {k}/{len(folders)} folders")

    # align to the common length (should be 11)
    L = min(min(len(r) for r in aes_rows), min(len(r) for r in drift_rows))
    A = np.array([r[:L] for r in aes_rows], float)          # (N, L) independent aesthetic
    D = np.array([r[:L] for r in drift_rows], float)        # (N, L) drift vs initial
    S = np.array([r[:L] for r in self_rows], float)         # (N, L) persona self-score

    aes_delta = A[:, -1] - A[:, 0]
    self_delta = S[:, -1] - S[:, 0]
    valid = ~np.isnan(self_delta)
    self_vs_indep = float(np.corrcoef(self_delta[valid], aes_delta[valid])[0, 1]) if valid.sum() > 2 else float("nan")

    rep = {
        "n_images": int(A.shape[0]), "n_iters": int(L - 1), "device": args.device,
        "independent_aesthetic": {
            "mean_trajectory": A.mean(0).round(4).tolist(),
            "start_mean": float(A[:, 0].mean()), "end_mean": float(A[:, -1].mean()),
            "delta_mean": float(aes_delta.mean()), "delta_ci": _boot_ci(aes_delta),
            "pct_improved": float((aes_delta > 0).mean()),
        },
        "drift_vs_initial": {
            "mean_trajectory": D.mean(0).round(4).tolist(),
            "end_mean": float(D[:, -1].mean()), "end_ci": _boot_ci(D[:, -1]),
        },
        "self_vs_independent": {
            "corr_delta": self_vs_indep,
            "note": "does the persona's self-score delta track the neutral aesthetic delta?",
        },
    }

    figs = os.path.join(OUT_DIR, "figs"); os.makedirs(figs, exist_ok=True)
    it = np.arange(L)

    # Fig 1: independent aesthetic trajectory (the non-circular one)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ci = np.array([_boot_ci(A[:, k]) for k in it])
    ax.plot(it, A.mean(0), "-o", color="#d62728", lw=2, label="independent LAION aesthetic")
    ax.fill_between(it, ci[:, 0], ci[:, 1], color="#d62728", alpha=0.2)
    ia = rep["independent_aesthetic"]
    ax.set_xlabel("refinement iteration"); ax.set_ylabel("held-out aesthetic score")
    ax.set_title(f"C4 self-refine — INDEPENDENT aesthetic vs iteration\n"
                 f"Δ {ia['delta_mean']:+.3f} CI[{ia['delta_ci'][0]:+.3f},{ia['delta_ci'][1]:+.3f}] "
                 f"· {ia['pct_improved']*100:.0f}% improved")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(figs, "c4sr_indep_aesthetic.png"), dpi=130); plt.close(fig)

    # Fig 2: drift vs initial (identity degradation of the compounding loop)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(it, D.mean(0), "-o", color="#1f77b4", lw=2)
    ax.axhline(0.78, ls="--", c="k", lw=0.8, label="c4_refine commit cap (0.78)")
    ax.set_xlabel("refinement iteration"); ax.set_ylabel("identity similarity to initial (DINOv2)")
    ax.set_title(f"C4 self-refine — identity drift over compounding edits\n"
                 f"end similarity {rep['drift_vs_initial']['end_mean']:.3f}")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(figs, "c4sr_drift.png"), dpi=130); plt.close(fig)

    # Fig 3: self-score delta vs independent delta (is the self-score trustworthy?)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(self_delta[valid], aes_delta[valid], s=22, alpha=0.6, color="#2ca02c", edgecolor="w")
    ax.axhline(0, c="k", lw=0.6); ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("persona self-score Δ"); ax.set_ylabel("independent aesthetic Δ")
    ax.set_title(f"C4 self-refine — self-score vs neutral metric\n"
                 f"corr = {self_vs_indep:+.2f} (≈0 ⇒ self-score wasn't tracking real quality)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c4sr_self_vs_indep.png"), dpi=130); plt.close(fig)

    rep["_figures"] = ["figs/c4sr_indep_aesthetic.png", "figs/c4sr_drift.png", "figs/c4sr_self_vs_indep.png"]
    p = write_json(rep, "c4_selfrefine_rescore.json")
    print(f"\nwrote {p} + 3 figs")
    print(f"independent aesthetic Δ {ia['delta_mean']:+.3f} CI{ia['delta_ci']} "
          f"({ia['pct_improved']*100:.0f}% improved)")
    print(f"end identity similarity {rep['drift_vs_initial']['end_mean']:.3f}")
    print(f"self-score vs independent corr {self_vs_indep:+.2f}")


if __name__ == "__main__":
    main()
