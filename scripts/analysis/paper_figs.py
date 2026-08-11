"""Composite main-text figures for the 6-page paper build.

The standalone per-experiment figures (b4_calibration, c1_separation, c3_*,
c4_headline, c4_trajectory) stay as they are and now live in the supplement.
This script packs the same numbers into three wide, low panels so the main text
fits the page limit without losing a claim:

  * pf_audience.png   — (left) calibration  (middle) C1 between-group
                        separation  (right) C3 panel-size curve
  * pf_autopolish.png — (a) best-so-far trajectory  (b) gain vs identity
  * pf_qualitative.png — tight 2-row source/edit grid
  * pf_progression.png — 2-row best-so-far progression across refinement steps

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
from c4_qualitative import (ROW_SWAP, _cell_path, _final_best, _source_path,
                            apply_row_order, load_c4)
from c4_progression import CHECKPOINTS
from c4_progression import _cell as _prog_cell
from c4_trajectory import CONDITIONS, _best_matrix, _boot_ci, _finals
from common import REPO

RES = os.path.join(REPO, "results")
PAPER = os.path.join(REPO, "docs", "paper")
# Each venue keeps its own figs/ (page limits and templates differ, so sizes are
# re-tuned per venue); --figs points the build at a different venue directory.
FIGS = os.path.join(PAPER, "neurips_creative_ai", "figs")
DSS = ["PARA", "EVA", "LAPIS"]
C4LABELS = {"static": "static string", "blind": "blind VLM", "society": "AutoPolish",
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
    # 1.42in, down from 1.56in: with no xlabel under the right panel the axes
    # keep their old drawing height (~1.21in) while the figure gets ~0.14in
    # shorter on the page, which is the text line this reclaims.
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(5.5, 1.42))
    x = np.arange(3)

    # No panel titles: the caption addresses the panels by position
    # (left / middle / right), so nothing is drawn on top of the axes.

    # (left) calibration: group MAE raw -> calibrated, against the population prior.
    # Raw bars carry the dataset hue (PARA orange, EVA blue, LAPIS green), so the
    # dataset palette reads the same way here as in the middle panel; calibrated
    # is gray and the population prior is the dashed black reference line.
    w = 0.34
    raw = [cal[d]["raw"]["group_mae"] for d in DSS]
    cald = [cal[d]["calibrated"]["group_mae"] for d in DSS]
    prior = [cal[d]["calibrated"]["population_prior_group_mae"] for d in DSS]
    a1.bar(x - w / 2, raw, w, color=[theme.DATASET[d] for d in DSS], zorder=3)
    a1.bar(x + w / 2, cald, w, color=theme.NEUTRAL, zorder=3)
    for xi, p in zip(x, prior):
        a1.plot([xi - 0.46, xi + 0.46], [p, p], color=theme.INK, lw=1.4,
                ls=(0, (3, 1.6)), zorder=4)
    a1.set_xticks(x)
    a1.set_xticklabels(DSS)
    a1.set_ylabel("group error (MAE)")
    a1.set_ylim(0, max(raw) * 1.45)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    rawkey = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    a1.legend([rawkey, Patch(facecolor=theme.NEUTRAL),
               Line2D([0], [0], color=theme.INK, lw=1.4, ls=(0, (3, 1.6)))],
              ["raw", "calib.", "prior"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper center", ncol=3, columnspacing=0.7, handlelength=1.6,
              handletextpad=0.35, borderpad=0.1, fontsize=5.5)

    # (middle) C1 between-group separation, persona panel vs no-persona control
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
              loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1,
              labelspacing=0.25, handletextpad=0.35, borderpad=0.1, fontsize=5.5)

    # (right) C3 panel-size curve on generated images
    nc = c3["aggregation"]["n_curve"]
    ns = sorted(int(k) for k in nc)
    ys = [nc[str(n)] for n in ns]
    maj = c3["aggregation"]["aggregate_acc_majority"]
    a3.axhline(maj, ls="--", lw=0.9, color=theme.NEUTRAL, zorder=2)
    a3.text(ns[-1], maj - 0.003, "majority prior", ha="right", va="top",
            fontsize=5.5, color=theme.MUTED)
    a3.plot(ns, ys, "-o", ms=3, color=theme.PRIMARY, zorder=3)
    # names the corpus: the other two panels label their datasets on the x-axis,
    # this one has panel size there, so the dataset has to be said somewhere
    a3.legend([Line2D([0], [0], color=theme.PRIMARY, marker="o", ms=3, lw=1.2)],
              ["Rapidata"], loc="upper left", handlelength=1.6,
              handletextpad=0.35, borderpad=0.1, fontsize=5.5)
    # above the point, not below it: the strip under the curve now carries the
    # axis name, which is what keeps this panel the same height as the other two
    a3.annotate(f"{ys[0]:.3f}", (ns[0], ys[0]), textcoords="offset points",
                xytext=(3, 3), fontsize=5.5, color=theme.INK)
    a3.annotate(f"{ys[-1]:.3f}", (ns[-1], ys[-1]), textcoords="offset points",
                xytext=(-2, 4), fontsize=5.5, color=theme.INK, ha="right")
    a3.set_xscale("log")
    a3.set_xticks(ns)
    a3.set_xticklabels([str(n) for n in ns])
    a3.minorticks_off()
    # The axis name goes inside the axes rather than under the tick row: an
    # xlabel here would add a band of height that the other two panels do not
    # have, making the whole figure one text line taller on the page.
    a3.text(0.5, 0.015, "panel size $N$", transform=a3.transAxes,
            ha="center", va="bottom", fontsize=6, color=theme.MUTED)
    a3.set_ylabel("agreement w/ crowd")
    a3.set_ylim(min(min(ys), maj) - 0.016, max(ys) + 0.018)
    a3.grid(True, axis="y")
    a3.set_axisbelow(True)

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

    # Main-text figure: drawn at the printed width (5.5in) like fig_audience, so
    # the point sizes below are the ones the reader sees. No panel titles, and
    # single-line y-labels: the caption addresses the panels by position, and
    # every row of height here is a row of text the 6-page budget loses.
    print_size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.42),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    # (left) best-so-far trajectory with bootstrap CI bands
    for c in present:
        M, _, steps = _best_matrix(data[c])
        mean = M.mean(0)
        ci = np.array([_boot_ci(M[:, s]) for s in range(M.shape[1])])
        a1.plot(steps, mean, "-o", ms=2.5, color=theme.C4[c], label=C4LABELS[c], zorder=3)
        a1.fill_between(steps, ci[:, 0], ci[:, 1], color=theme.C4[c], alpha=0.09, zorder=1)
    a1.set_xlabel("refinement step")
    a1.set_ylabel("best-so-far held-out score")
    a1.legend(loc="lower right", fontsize=5.5, borderpad=0.15, labelspacing=0.22,
              handlelength=1.4, handletextpad=0.35)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)

    # (right) per-image final gain vs identity similarity of the committed best.
    # One dot per image per condition. It repeats the left panel's color key
    # rather than borrowing it: readers do not carry a line-chart legend across
    # into a dense scatter, so this panel states its own.
    for c in present:
        f = _finals(data[c])
        xs = [v["drift_final"] for v in f.values()]
        ys = [v["gain"] for v in f.values()]
        a2.scatter(xs, ys, s=4.5, alpha=0.72, color=theme.C4[c], linewidth=0, zorder=3)
    a2.axvline(drift_cap, ls=(0, (3, 1.6)), lw=1.0, color=theme.INK, zorder=4)
    # headroom so the legend sits over empty plot rather than over the points
    ytop = a2.get_ylim()[1]
    a2.set_ylim(a2.get_ylim()[0], ytop * 1.20)
    a2.text(drift_cap + 0.008, ytop * 0.86, f"drift cap ({drift_cap:g})",
            va="top", ha="left", fontsize=5.5, color=theme.MUTED)
    a2.legend(handles=[Line2D([0], [0], marker="o", ms=2.4, lw=0,
                              color=theme.C4[c], label=C4LABELS[c]) for c in present],
              loc="upper center", ncol=2, fontsize=5.5, borderpad=0.2,
              labelspacing=0.22, columnspacing=0.9, handlelength=1.0,
              handletextpad=0.3, framealpha=0.9)
    a2.set_xlabel("identity similarity of the committed best (DINOv2)")
    a2.set_ylabel("final held-out gain")
    a2.grid(True)
    a2.set_axisbelow(True)

    fig.tight_layout(w_pad=1.0, pad=0.25)
    p = os.path.join(FIGS, "pf_autopolish.png")
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 4 — tight qualitative grid
# --------------------------------------------------------------------------
def fig_qualitative(logs_dir: str, edits_dir: str, n_show: int = 2, skip: int = 0,
                    row_offset: int = 1, out_name: str = "pf_qualitative.png") -> str:
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
            if _cell_path(edits_dir, c, iid, finals[c].loc[iid]["best_path"]) is None:
                return False
        return True

    # Gather far enough down the ranking to apply the shared row-order override,
    # then keep the first n_show rows -- so this figure stays the top rows of the
    # supplement's full grid however that order is arranged.
    n_gather = max(n_show + row_offset, max(ROW_SWAP) + 1)
    picks: list[str] = []
    for iid in list(gain.index)[skip:]:
        if complete(iid):
            picks.append(iid)
        if len(picks) >= n_gather:
            break
    picks = apply_row_order(picks)[row_offset:row_offset + n_show]

    start = (data["society"][data["society"]["step"] == 0]
             .set_index("image_id")["best_obj"].to_dict())

    print_size()
    cols = ["source"] + present
    # Row height follows the images' own aspect ratio, so rows sit close
    # together instead of being separated by a band of unused axes.
    cell_w = 5.5 / len(cols)
    aspects = []
    for iid in picks:
        with Image.open(_source_path(edits_dir, iid)) as im:
            aspects.append(im.height / im.width)
    row_h = cell_w * max(aspects) + 0.13   # + label line
    # Drawn at the final printed width so the per-cell labels stay legible.
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(5.5, row_h * len(picks)),
                             gridspec_kw=dict(wspace=0.02, hspace=0.10))
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
                path = _cell_path(edits_dir, col, img_id,
                                  finals[col].loc[img_id]["best_path"])
                score = float(finals[col].loc[img_id]["best_obj"])
                name = QLABELS[col]
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
            # every row carries the full label: condition name and score. Our
            # method is marked with weight rather than hue, so the figure keeps
            # working in grayscale and for colorblind readers.
            label = f"{name}  {score:.2f}"
            weight = "bold" if col == "society" else "normal"
            ax.set_title(label, fontsize=6, pad=1.6, fontweight=weight,
                         color=theme.INK)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 5 — the refinement loop over time (2 rows)
# --------------------------------------------------------------------------
# Which rows of the supplement's full 7-row progression grid to lift into the
# main text. Same ranking, so the main figure is a subset of the supplement's.
PROG_ROWS = (2, 6)


def fig_progression(logs_dir: str, edits_dir: str,
                    out_name: str = "pf_progression.png") -> str:
    df = load_c4("society", logs_dir)
    scored = []
    for iid, g in df.groupby("image_id"):
        g = g.sort_values("step")
        cells = [_prog_cell(g, st, edits_dir, iid) for st in CHECKPOINTS]
        if any(pt is None or not os.path.exists(pt) for pt, _, _ in cells):
            continue
        distinct = len({os.path.basename(pt) for pt, _, _ in cells})
        gain = cells[-1][1] - cells[0][1]
        scored.append((distinct, gain, iid, cells))
    # most visible progression first (distinct checkpoints), then largest gain
    scored.sort(key=lambda r: (-r[0], -r[1]))
    picks = [scored[i] for i in PROG_ROWS if i < len(scored)]

    print_size()
    colnames = ["source"] + [f"step {st}" for st in CHECKPOINTS[1:]]
    # Row height follows the images' own aspect ratio. A fixed height leaves a
    # band of dead space under every landscape row, which on a 6-page budget is
    # whitespace the paper cannot afford.
    cell_w = 5.5 / len(CHECKPOINTS)
    aspects = []
    for _, _, _iid, cells in picks:
        with Image.open(cells[0][0]) as im:
            aspects.append(im.height / im.width)
    row_h = cell_w * max(aspects) + 0.13   # + label line
    # drawn at the printed width so the per-cell labels stay legible
    fig, axes = plt.subplots(len(picks), len(CHECKPOINTS),
                             figsize=(5.5, row_h * len(picks)),
                             gridspec_kw=dict(wspace=0.02, hspace=0.16))
    axes = np.atleast_2d(axes)
    for r, (_, _, _iid, cells) in enumerate(picks):
        for c, (path, score, _) in enumerate(cells):
            ax = axes[r, c]
            ax.axis("off")
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
            # every row carries the full label, as in fig_qualitative
            ax.set_title(f"{colnames[c]}  {score:.2f}", fontsize=6, pad=1.6,
                         color=theme.INK)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Composite main-text paper figures.")
    ap.add_argument("--c4-root", default=os.path.join(REPO, "data", "results", "c4_run2"))
    ap.add_argument("--n-show", type=int, default=2)
    ap.add_argument("--figs", default=FIGS, help="output figure dir (one per venue)")
    args = ap.parse_args()
    FIGS = args.figs
    logs = os.path.join(args.c4_root, "logs")
    edits = os.path.join(args.c4_root, "edits")

    os.makedirs(FIGS, exist_ok=True)
    print("wrote", fig_audience())
    print("wrote", fig_autopolish(logs))
    print("wrote", fig_qualitative(logs, edits, n_show=args.n_show))
    print("wrote", fig_progression(logs, edits))
