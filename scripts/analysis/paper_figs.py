"""Composite main-text figures for the 6-page paper build.

The standalone per-experiment figures (b4_calibration, c1_separation, c3_*,
c4_headline, c4_trajectory) stay as they are and now live in the supplement.
This script packs the same numbers into three wide, low panels so the main text
fits the page limit without losing a claim:

  * pf_audience.png   — (a) calibration  (b) C1 between-group separation
                        (c) C3 panel-size curve
  * pf_autopolish.png — (a) best-so-far trajectory  (b) gain vs identity
  * pf_qualitative.png — tight 3-row source/edit grid

Pure re-analysis (no GPU). Run from ``scripts/analysis/``::

    python paper_figs.py --c4-root ../../data/results/c4_run2
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import theme
from c4_qualitative import LABELS as QLABELS
from c4_qualitative import _final_best, _source_path, load_c4
from c4_trajectory import CONDITIONS, _best_matrix, _boot_ci, _finals
from common import REPO

RES = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "docs", "paper", "figs")
DSS = ["PARA", "EVA", "LAPIS"]
C4LABELS = {"static": "static string", "blind": "blind VLM", "society": "society",
            "reward_only": "reward-only (oracle)"}


def load(name):
    with open(os.path.join(RES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def print_size(base: float = 6.5) -> None:
    """Type sized for the printed page.

    The main-text figures are drawn at their final physical width (5.5in, the
    NeurIPS text block) and included at ``width=\\linewidth``, so every point
    size here is the point size the reader actually sees. Drawing wide and
    scaling down is what makes composite figures unreadable in print.
    """
    theme.apply()
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


# --------------------------------------------------------------------------
# Figure 2 — the synthetic audience is faithful (3 panels)
# --------------------------------------------------------------------------
def fig_audience() -> str:
    cal = load("calibration")
    c1 = load("c1_separation")
    c3 = load("c3")

    print_size()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(5.5, 1.56))
    x = np.arange(3)

    def panel_tag(ax, tag):
        ax.set_title(tag, fontsize=7, loc="left", pad=2.5)

    # (a) calibration: group MAE raw -> calibrated, against the population prior
    w = 0.34
    raw = [cal[d]["raw"]["group_mae"] for d in DSS]
    cald = [cal[d]["calibrated"]["group_mae"] for d in DSS]
    prior = [cal[d]["calibrated"]["population_prior_group_mae"] for d in DSS]
    a1.bar(x - w / 2, raw, w, color=theme.RAW, zorder=3)
    a1.bar(x + w / 2, cald, w, color=theme.CAL, zorder=3)
    for xi, p in zip(x, prior):
        a1.plot([xi - 0.46, xi + 0.46], [p, p], color=theme.PRIOR, lw=1.4, zorder=4)
    a1.set_xticks(x)
    a1.set_xticklabels(DSS)
    a1.set_ylabel("group error (MAE)")
    a1.set_ylim(0, max(raw) * 1.45)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    a1.legend(handles=[Patch(facecolor=theme.RAW, label="raw"),
                       Patch(facecolor=theme.CAL, label="calib."),
                       Line2D([0], [0], color=theme.PRIOR, lw=1.4, label="prior")],
              loc="upper center", ncol=3, columnspacing=0.7, handlelength=1.0,
              handletextpad=0.35, borderpad=0.1, fontsize=5.5)
    panel_tag(a1, "(a) calibration")

    # (b) C1 between-group separation, persona panel vs no-persona control
    w = 0.36
    full = [c1[d]["overall"]["full_separation"]["corr"] for d in DSS]
    blind = [c1[d]["overall"]["blind_separation"]["corr"] for d in DSS]
    err = np.array([[f - c1[d]["overall"]["full_separation"]["ci95"][0],
                     c1[d]["overall"]["full_separation"]["ci95"][1] - f]
                    for f, d in zip(full, DSS)]).T
    a2.bar(x - w / 2, full, w, yerr=err, capsize=1.6,
           error_kw=dict(ecolor=theme.INK, lw=0.7, capthick=0.7),
           color=[theme.DATASET[d] for d in DSS], zorder=3)
    a2.bar(x + w / 2, blind, w, color=theme.NEUTRAL, zorder=3)
    a2.axhline(0, color=theme.MUTED, lw=0.6)
    a2.set_xticks(x)
    a2.set_xticklabels(DSS)
    a2.set_ylabel("group separation $r$")
    a2.set_ylim(-0.135, 0.275)
    a2.grid(True, axis="y")
    a2.set_axisbelow(True)
    key = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    a2.legend([key, Patch(facecolor=theme.NEUTRAL)], ["persona panel", "no persona"],
              handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=1.6,
              loc="upper center", ncol=2, columnspacing=0.7, handletextpad=0.35,
              borderpad=0.1, fontsize=5.5)
    panel_tag(a2, "(b) between-group")

    # (c) C3 panel-size curve on generated images
    nc = c3["aggregation"]["n_curve"]
    ns = sorted(int(k) for k in nc)
    ys = [nc[str(n)] for n in ns]
    maj = c3["aggregation"]["aggregate_acc_majority"]
    a3.axhline(maj, ls="--", lw=0.9, color=theme.NEUTRAL, zorder=2)
    a3.text(ns[-1], maj - 0.003, "majority prior", ha="right", va="top",
            fontsize=5.5, color=theme.MUTED)
    a3.plot(ns, ys, "-o", ms=3, color=theme.PRIMARY, zorder=3)
    a3.annotate(f"{ys[0]:.3f}", (ns[0], ys[0]), textcoords="offset points",
                xytext=(3, -7), fontsize=5.5, color=theme.INK)
    a3.annotate(f"{ys[-1]:.3f}", (ns[-1], ys[-1]), textcoords="offset points",
                xytext=(-2, 4), fontsize=5.5, color=theme.INK, ha="right")
    a3.set_xscale("log")
    a3.set_xticks(ns)
    a3.set_xticklabels([str(n) for n in ns])
    a3.minorticks_off()
    a3.set_xlabel("panel size $N$")
    a3.set_ylabel("agreement w/ crowd")
    a3.set_ylim(min(min(ys), maj) - 0.016, max(ys) + 0.018)
    a3.grid(True, axis="y")
    a3.set_axisbelow(True)
    panel_tag(a3, "(c) panel size")

    fig.tight_layout(w_pad=0.9, pad=0.25)
    p = os.path.join(FIGS, "pf_audience.png")
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 3 — AutoPolish quantitative (2 panels)
# --------------------------------------------------------------------------
def fig_autopolish(logs_dir: str, drift_cap: float = 0.78) -> str:
    data = {c: load_c4(c, logs_dir) for c in CONDITIONS}
    present = [c for c in CONDITIONS if len(data[c])]

    theme.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 3.1),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    # (a) best-so-far trajectory with bootstrap CI bands
    for c in present:
        M, _, steps = _best_matrix(data[c])
        mean = M.mean(0)
        ci = np.array([_boot_ci(M[:, s]) for s in range(M.shape[1])])
        a1.plot(steps, mean, "-o", ms=4.5, color=theme.C4[c], label=C4LABELS[c], zorder=3)
        a1.fill_between(steps, ci[:, 0], ci[:, 1], color=theme.C4[c], alpha=0.09, zorder=1)
    a1.set_xlabel("refinement step")
    a1.set_ylabel("best-so-far held-out\naesthetic score")
    a1.legend(loc="lower right", fontsize=8.5, borderpad=0.2, labelspacing=0.3)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    a1.set_title("(a)  refinement converges by step 3–4", fontsize=9.5)

    # (b) per-image final gain vs identity similarity of the committed best.
    # Colors are keyed by panel (a)'s legend, so this panel carries none.
    for c in present:
        f = _finals(data[c])
        xs = [v["drift_final"] for v in f.values()]
        ys = [v["gain"] for v in f.values()]
        a2.scatter(xs, ys, s=17, alpha=0.72, color=theme.C4[c], linewidth=0, zorder=3)
    a2.axvline(drift_cap, ls="--", lw=1.4, color=theme.INK, zorder=4)
    a2.text(drift_cap + 0.008, a2.get_ylim()[1] * 0.88, f"drift cap ({drift_cap:g})",
            va="top", ha="left", fontsize=8.5, color=theme.MUTED)
    a2.set_xlabel("identity similarity of the committed best (DINOv2)")
    a2.set_ylabel("final held-out\naesthetic gain")
    a2.grid(True)
    a2.set_axisbelow(True)
    a2.set_title("(b)  gains come with identity preserved", fontsize=9.5)

    fig.tight_layout(w_pad=1.6)
    p = os.path.join(FIGS, "pf_autopolish.png")
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 4 — tight qualitative grid
# --------------------------------------------------------------------------
def fig_qualitative(logs_dir: str, edits_dir: str, n_show: int = 3, skip: int = 0,
                    out_name: str = "pf_qualitative.png") -> str:
    data = {c: load_c4(c, logs_dir) for c in CONDITIONS}
    present = [c for c in CONDITIONS if len(data[c])]
    finals = {c: _final_best(data[c]) for c in present}
    soc = finals["society"]["best_obj"]
    base = finals["static"]["best_obj"]
    common = soc.index.intersection(base.index)
    gain = (soc.loc[common] - base.loc[common]).sort_values(ascending=False)

    def complete(iid: str) -> bool:
        if not _source_path(edits_dir, iid):
            return False
        for c in present:
            if iid not in finals[c].index:
                return False
            bp = os.path.basename(str(finals[c].loc[iid]["best_path"]))
            if not os.path.exists(os.path.join(edits_dir, c, iid, bp)):
                return False
        return True

    picks: list[str] = []
    for iid in list(gain.index)[skip:]:
        if complete(iid):
            picks.append(iid)
        if len(picks) >= n_show:
            break

    start = (data["society"][data["society"]["step"] == 0]
             .set_index("image_id")["best_obj"].to_dict())

    print_size()
    cols = ["source"] + present
    # Drawn at the final printed width so the per-cell labels stay legible.
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(5.5, 0.98 * len(picks)),
                             gridspec_kw=dict(wspace=0.02, hspace=0.22))
    axes = np.atleast_2d(axes)
    for r, img_id in enumerate(picks):
        src = _source_path(edits_dir, img_id)
        for cc, col in enumerate(cols):
            ax = axes[r, cc]
            ax.axis("off")
            if col == "source":
                path, score = src, start.get(img_id, float("nan"))
                name = "source"
            else:
                bp = str(finals[col].loc[img_id]["best_path"])
                path = os.path.join(edits_dir, col, img_id, os.path.basename(bp))
                score = float(finals[col].loc[img_id]["best_obj"])
                name = QLABELS[col]
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
            # condition name only on the top row; the score on every cell
            label = f"{name}  {score:.2f}" if r == 0 else f"{score:.2f}"
            weight = "bold" if col == "society" else "normal"
            ax.set_title(label, fontsize=6, pad=1.6, fontweight=weight,
                         color=theme.PRIMARY if col == "society" else theme.INK)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Composite main-text paper figures.")
    ap.add_argument("--c4-root", default=os.path.join(REPO, "data", "results", "c4_run2"))
    ap.add_argument("--n-show", type=int, default=3)
    args = ap.parse_args()
    logs = os.path.join(args.c4_root, "logs")
    edits = os.path.join(args.c4_root, "edits")

    os.makedirs(FIGS, exist_ok=True)
    print("wrote", fig_audience())
    print("wrote", fig_autopolish(logs))
    print("wrote", fig_qualitative(logs, edits, n_show=args.n_show))
