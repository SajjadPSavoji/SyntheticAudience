"""C4 (self-refinement run) — analysis of the Qwen-Image-Edit persona loop.

Run under data/results/c4_selfrefine/: 100 PARA images, each refined for 10
iterations by its real annotator persona, which every step rates the current
image (aestheticScore 1-5) AND proposes an edit (applied by Qwen-Image-Edit).
The only logged numeric signal is the persona's OWN score trajectory.

IMPORTANT (scope): this is a *self-graded* loop — the same persona proposes the
edit and scores it (proposer == evaluator), there is no independent held-out
objective, and there is no society/blind/static condition. So a rising
trajectory would be self-report, not proof; and here it is essentially flat.
This script quantifies exactly that and diagnoses why (regression to the mean).

Run from scripts/analysis/. Writes results/c4_selfrefine.json + figures +
results/c4_selfrefine_summary.md. No GPU, no image reads.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, REPO, ensure_out, write_json

RUN = os.path.join(REPO, "data", "results", "c4_selfrefine")
RNG = np.random.default_rng(0)
N_BOOT = 2000


def _boot_ci(x, stat=np.mean, n=N_BOOT):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    d = stat(x[idx], axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def load():
    s = json.load(open(os.path.join(RUN, "summary.json")))
    imgs = s["images"]
    T = np.array([im["score_trajectory"] for im in imgs], float)   # (N, 11)
    # instruction repetition from the per-folder logs
    uniq = []
    for lg in glob.glob(os.path.join(RUN, "session*", "log.json")):
        its = json.load(open(lg))["iterations"]
        instr = [" ".join(it.get("edit_instruction", "").lower().split())
                 for it in its if it.get("edit_instruction")]
        if instr:
            uniq.append(len(set(instr)) / len(instr))
    return s["config"], T, np.array(uniq)


def analyze() -> dict:
    cfg, T, uniq = load()
    start, end = T[:, 0], T[:, -1]
    delta = end - start
    steps = np.diff(T, axis=1)

    rep = {
        "config": {k: cfg[k] for k in ("dataset", "vlm_model", "edit_model",
                                       "n_images", "n_iterations", "temperature")},
        "self_score": {
            "start_mean": float(start.mean()), "end_mean": float(end.mean()),
            "delta_mean": float(delta.mean()), "delta_ci": _boot_ci(delta),
            "delta_sd": float(delta.std()),
            "pct_improved": float((delta > 0).mean()),
            "pct_flat": float((delta == 0).mean()),
            "pct_declined": float((delta < 0).mean()),
            "mean_gain_when_improved": float(delta[delta > 0].mean()) if (delta > 0).any() else None,
            "mean_loss_when_declined": float(delta[delta < 0].mean()) if (delta < 0).any() else None,
        },
        "dynamics": {
            "mean_trajectory": T.mean(0).round(4).tolist(),
            "step_up": float((steps > 0).mean()), "step_flat": float((steps == 0).mean()),
            "step_down": float((steps < 0).mean()),
            "instr_unique_frac_mean": float(uniq.mean()) if len(uniq) else None,
        },
        "regression_to_mean": {
            "corr_start_delta": float(np.corrcoef(start, delta)[0, 1]),
            "mean_delta_by_start": {str(v): float(delta[start == v].mean())
                                    for v in sorted(set(start.tolist()))},
            "n_by_start": {str(v): int((start == v).sum()) for v in sorted(set(start.tolist()))},
        },
    }
    _figures(rep, T, start, delta)
    return rep


def _figures(rep, T, start, delta):
    figs = os.path.join(OUT_DIR, "figs"); os.makedirs(figs, exist_ok=True)
    it = np.arange(T.shape[1])

    # Fig 1: mean self-score trajectory + CI band + faint per-image lines
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in T:
        ax.plot(it, row, color="#bbbbbb", lw=0.4, alpha=0.35)
    m = T.mean(0)
    ci = np.array([_boot_ci(T[:, k]) for k in it])
    ax.plot(it, m, "-o", color="#d62728", lw=2, label="mean self-score")
    ax.fill_between(it, ci[:, 0], ci[:, 1], color="#d62728", alpha=0.2)
    ax.set_xlabel("refinement iteration")
    ax.set_ylabel("persona's own aesthetic score (1-5)")
    ax.set_title("C4 self-refinement — the self-score trajectory is essentially flat\n"
                 f"(mean {rep['self_score']['start_mean']:.2f} -> {rep['self_score']['end_mean']:.2f}, "
                 f"delta {rep['self_score']['delta_mean']:+.2f} "
                 f"CI[{rep['self_score']['delta_ci'][0]:+.2f},{rep['self_score']['delta_ci'][1]:+.2f}])")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c4sr_trajectory.png"), dpi=130); plt.close(fig)

    # Fig 2: delta distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(delta.min() - 0.25, delta.max() + 0.75, 0.5)
    ax.hist(delta, bins=bins, color="#1f77b4", edgecolor="w")
    ax.axvline(0, c="k", lw=1)
    sc = rep["self_score"]
    ax.set_xlabel("final − initial self-score")
    ax.set_ylabel("images")
    ax.set_title(f"C4 self-refinement — outcome distribution\n"
                 f"improved {sc['pct_improved']*100:.0f}% · flat {sc['pct_flat']*100:.0f}% · "
                 f"declined {sc['pct_declined']*100:.0f}%")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c4sr_delta_hist.png"), dpi=130); plt.close(fig)

    # Fig 3: regression-to-the-mean diagnostic (mean delta by start score)
    rtm = rep["regression_to_mean"]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(float(k) for k in rtm["mean_delta_by_start"])
    ys = [rtm["mean_delta_by_start"][str(x)] for x in xs]
    ns = [rtm["n_by_start"][str(x)] for x in xs]
    ax.bar([str(x) for x in xs], ys, color=["#2ca02c" if y > 0 else "#d62728" for y in ys])
    ax.axhline(0, c="k", lw=0.8)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + (0.05 if y >= 0 else -0.12), f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("initial self-score")
    ax.set_ylabel("mean change over 10 iterations")
    ax.set_title(f"C4 self-refinement — gains are REGRESSION TO THE MEAN\n"
                 f"corr(start, delta) = {rtm['corr_start_delta']:+.2f} "
                 f"(low starts rise, high starts fall, all -> ~3)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c4sr_regression.png"), dpi=130); plt.close(fig)
    rep["_figures"] = ["figs/c4sr_trajectory.png", "figs/c4sr_delta_hist.png", "figs/c4sr_regression.png"]


def _table(rep) -> str:
    sc, dy, rtm = rep["self_score"], rep["dynamics"], rep["regression_to_mean"]
    L = ["# C4 self-refinement — results summary", "",
         "*Self-graded loop (proposer == evaluator); no independent objective; no society/blind/static condition.*", "",
         "| metric | value |", "|---|---|",
         f"| images × iterations | {rep['config']['n_images']} × {rep['config']['n_iterations']} |",
         f"| editor / VLM | {rep['config']['edit_model']} / {rep['config']['vlm_model']} (temp {rep['config']['temperature']}) |",
         f"| mean self-score start → end | {sc['start_mean']:.2f} → {sc['end_mean']:.2f} |",
         f"| **mean Δ self-score** | **{sc['delta_mean']:+.3f}  CI[{sc['delta_ci'][0]:+.3f}, {sc['delta_ci'][1]:+.3f}]** (CI includes 0) |",
         f"| improved / flat / declined | {sc['pct_improved']*100:.0f}% / {sc['pct_flat']*100:.0f}% / {sc['pct_declined']*100:.0f}% |",
         f"| step deltas up / flat / down | {dy['step_up']*100:.0f}% / {dy['step_flat']*100:.0f}% / {dy['step_down']*100:.0f}% |",
         f"| unique edit-instructions (frac of 10) | {dy['instr_unique_frac_mean']:.2f} |",
         f"| **corr(start, Δ)** — regression to mean | **{rtm['corr_start_delta']:+.2f}** |",
         "", "**Mean Δ by initial score (regression-to-mean signature):**", "",
         "| start | n | mean Δ |", "|---|---|---|"]
    for k in sorted(rtm["mean_delta_by_start"], key=float):
        L.append(f"| {k} | {rtm['n_by_start'][k]} | {rtm['mean_delta_by_start'][k]:+.2f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ensure_out()
    rep = analyze()
    p = write_json(rep, "c4_selfrefine.json")
    with open(os.path.join(OUT_DIR, "c4_selfrefine_summary.md"), "w") as f:
        f.write(_table(rep))
    sc = rep["self_score"]; rtm = rep["regression_to_mean"]
    print(f"wrote {p} + c4_selfrefine_summary.md + 3 figs")
    print(f"mean Δ self-score {sc['delta_mean']:+.3f} CI[{sc['delta_ci'][0]:+.3f},{sc['delta_ci'][1]:+.3f}] "
          f"(includes 0 => NOT significant)")
    print(f"improved/flat/declined: {sc['pct_improved']*100:.0f}/{sc['pct_flat']*100:.0f}/{sc['pct_declined']*100:.0f}%")
    print(f"corr(start,Δ) = {rtm['corr_start_delta']:+.2f}  => gains are regression to the mean")


if __name__ == "__main__":
    main()
