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
from matplotlib import patheffects
from matplotlib import transforms as mtransforms
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
DPI = 400   # photo grids dominate PDF size; --dpi trades size against print quality
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
    a3.axhline(maj, ls="--", lw=0.9, color=theme.REF, zorder=2)
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
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# ACCV variants. That venue splits the old three-panel audience section into
# one section per claim, so each needs a figure it can reference on its own
# page; the composites above stay as they are for NeurIPS and WACV, which
# share this figs/ directory and whose captions still say left/middle/right.
# --------------------------------------------------------------------------
def _err(vals, cis):
    """Asymmetric error-bar half-widths from point estimates and 95% CIs."""
    vals = np.asarray(vals, dtype=float)
    lo = np.array([c[0] for c in cis], dtype=float)
    hi = np.array([c[1] for c in cis], dtype=float)
    return np.vstack([vals - lo, hi - vals])


ERRKW = dict(ecolor=theme.INK, elinewidth=0.6, capsize=1.4, capthick=0.6)


def fig_persona() -> str:
    """Persona steerability: calibration, then the two text-level effects.

    The calibration panel moves here from ``pf_audience`` because the ACCV
    draft states the calibration result in the persona section, and a figure
    should sit in the section that cites it first.
    """
    cal = load("calibration")
    rat = load("rationale")

    print_size()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(5.5, 0.96))
    x = np.arange(3)

    # (left) calibration: group MAE raw -> calibrated, against the population
    # prior. Raw bars carry the dataset hue, calibrated is gray, prior is the
    # dashed reference line -- the same encoding as the panels that follow.
    w = 0.34
    raw = [cal[d]["raw"]["group_mae"] for d in DSS]
    cald = [cal[d]["calibrated"]["group_mae"] for d in DSS]
    prior = [cal[d]["calibrated"]["population_prior_group_mae"] for d in DSS]
    a1.bar(x - w / 2, raw, w, color=[theme.DATASET[d] for d in DSS], zorder=3,
           yerr=_err(raw, [cal[d]["raw"]["group_mae_ci95"] for d in DSS]), error_kw=ERRKW)
    a1.bar(x + w / 2, cald, w, color=theme.NEUTRAL, zorder=3,
           yerr=_err(cald, [cal[d]["calibrated"]["group_mae_ci95"] for d in DSS]), error_kw=ERRKW)
    for xi, p in zip(x, prior):
        a1.plot([xi - 0.46, xi + 0.46], [p, p], color=theme.INK, lw=1.4,
                ls=(0, (3, 1.6)), zorder=4)
    a1.set_xticks(x)
    a1.set_xticklabels(DSS)
    a1.set_ylabel("group error\n(MAE)")
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

    # (middle) a probe recovers the rater's attribute from the rationale text
    # alone. The persona-blind control sitting on the chance line is the whole
    # point, so the line is drawn over the bars rather than under them.
    w = 0.36
    # rater-split AUC (mean over folds) so the bars and their bootstrap CIs come
    # from the same estimator; within 0.003 of the random-split values.
    full = [rat[d]["auc_full_rater_oof"] for d in DSS]
    blind = [rat[d]["auc_blind_rater_oof"] for d in DSS]
    a2.bar(x - w / 2, full, w, color=[theme.DATASET[d] for d in DSS], zorder=3,
           yerr=_err(full, [rat[d]["auc_full_rater_oof_ci95"] for d in DSS]), error_kw=ERRKW)
    a2.bar(x + w / 2, blind, w, color=theme.NEUTRAL, zorder=3,
           yerr=_err(blind, [rat[d]["auc_blind_rater_oof_ci95"] for d in DSS]), error_kw=ERRKW)
    a2.axhline(0.5, ls=(0, (3, 1.6)), lw=1.0, color=theme.INK, zorder=4)
    a2.set_xticks(x)
    a2.set_xticklabels(DSS)
    a2.set_ylabel("attribute AUC\nfrom text")
    # headroom for the legend: the tallest bar is ~0.70
    a2.set_ylim(0.45, 0.79)
    a2.grid(True, axis="y")
    a2.set_axisbelow(True)
    key = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    a2.legend([key, Patch(facecolor=theme.NEUTRAL),
               Line2D([0], [0], color=theme.INK, lw=1.0, ls=(0, (3, 1.6)))],
              ["persona", "blind", "chance"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper center", ncol=3, columnspacing=0.5, handlelength=1.2,
              handletextpad=0.3, borderpad=0.1, fontsize=5.5)

    # (right) how many of the rationales are distinct. Same two-bar encoding as
    # the middle panel, with its own persona/blind key (no chance line here), and
    # headroom so the key clears the tall PARA bar.
    fd = [rat[d]["rationale_diversity_full"] for d in DSS]
    bd = [rat[d]["rationale_diversity_blind"] for d in DSS]
    a3.bar(x - w / 2, fd, w, color=[theme.DATASET[d] for d in DSS], zorder=3,
           yerr=_err(fd, [rat[d]["rationale_diversity_full_ci95"] for d in DSS]), error_kw=ERRKW)
    a3.bar(x + w / 2, bd, w, color=theme.NEUTRAL, zorder=3,
           yerr=_err(bd, [rat[d]["rationale_diversity_blind_ci95"] for d in DSS]), error_kw=ERRKW)
    a3.set_xticks(x)
    a3.set_xticklabels(DSS)
    a3.set_ylabel("distinct\ncomments\nper rater")
    a3.set_ylim(0, max(fd) * 1.45)
    a3.grid(True, axis="y")
    a3.set_axisbelow(True)
    a3.legend([key, Patch(facecolor=theme.NEUTRAL)], ["persona", "blind"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper left", ncol=2, columnspacing=0.7, handlelength=1.6,
              handletextpad=0.35, borderpad=0.1, fontsize=5.5)

    fig.tight_layout(w_pad=0.9, pad=0.25)
    p = os.path.join(FIGS, "pf_persona.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight",
                pad_inches=0.02)  # thin margin keeps the caption close
    plt.close(fig)
    return p




def fig_steer(width: float = 5.5, height: float = 0.92) -> str:
    """Score-level steerability, redrawn at text width for the ACCV main text.

    Same numbers as ``steerability.plot`` (b1_steerability.png, now superseded in
    the paper): one point per (attribute, level) group, placed by how far that
    group's real ratings depart from the per-image crowd mean (x) and how far the
    VLM's ratings depart under that group's persona (y). Drawn wide and low so it
    can sit under ``pf_persona`` at full width without doubling its height.
    """
    from steerability import DATASETS, steer_dataset

    print_size()
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    for k, (ax, ds) in enumerate(zip(axes, DSS)):
        rep = steer_dataset(ds, DATASETS[ds])
        cells = rep["_cells"]
        xe = np.array([c["empirical_effect"] for c in cells])
        ye = np.array([c["vlm_effect"] for c in cells])
        n = np.array([c["n"] for c in cells], dtype=float)
        lim = max(np.abs(xe).max(), np.abs(ye).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], ls=(0, (3, 1.6)), color=theme.REF, lw=0.8, zorder=1)
        ax.axhline(0, color=theme.GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=theme.GRID, lw=0.6, zorder=0)
        ax.scatter(xe, ye, s=np.sqrt(n) * 0.55, alpha=0.75, color=theme.DATASET[ds],
                   edgecolor=theme.rim(theme.DATASET[ds]), linewidth=0.3, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # r sits on the agreement line it is measured against. Rotation is given
        # in data space, so the text follows the diagonal however wide the panel;
        # it ends short of the corner and a white halo keeps it clear of the dots.
        ax.text(0.88 * lim, 0.88 * lim, f"r = {rep['steerability_corr']:+.2f}",
                transform=mtransforms.offset_copy(ax.transData, fig=fig, y=1.8, units="points"),
                rotation=45, rotation_mode="anchor", transform_rotates_text=True,
                ha="right", va="bottom", color=theme.INK, fontweight="bold",
                fontsize=plt.rcParams["axes.titlesize"] - 2, zorder=5,
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")])
        # The dataset key goes in the upper-left corner, which no panel uses.
        ax.legend([Line2D([0], [0], ls="", marker="o", markersize=4,
                          markerfacecolor=theme.DATASET[ds], markeredgecolor=theme.rim(theme.DATASET[ds]),
                          markeredgewidth=0.3)], [ds],
                  loc="upper left", frameon=False, handletextpad=0.2,
                  borderaxespad=0.2, borderpad=0.1)
        ax.set_xlabel("real group departure")
        if k == 0:
            ax.set_ylabel("VLM group\ndeparture")
        ax.grid(False)
    fig.tight_layout(w_pad=0.9, pad=0.25)
    p = os.path.join(FIGS, "pf_steer.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight",
                pad_inches=0.02)  # thin margin keeps the caption close
    plt.close(fig)
    return p


def fig_separation(width: float = 2.15, height: float = 0.88) -> str:
    """The between-group separation panel on its own, plus the cross-dataset average.

    Drawn narrow because the ACCV draft sets it beside the Holm table rather
    than beside another panel; ``width`` is the physical inches it occupies on
    the page, so the type size here is the type size in print. The last group is
    the mean of the three datasets (``_average`` in c1_separation.json, with a
    bootstrap CI over all three), drawn in dark gray and set off by a dashed rule.
    """
    c1 = load("c1_separation")
    avg = c1["_average"]

    print_size()
    fig, ax = plt.subplots(figsize=(width, height))
    labels = DSS + ["Average"]
    x = np.arange(len(labels))
    w = 0.36

    def series(kind):
        vals = [c1[d]["overall"][kind]["corr"] for d in DSS] + [avg[kind]["corr"]]
        cis = [c1[d]["overall"][kind]["ci95"] for d in DSS] + [avg[kind]["ci95"]]
        err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals, cis)]).T
        return vals, err

    full, ferr = series("full_separation")
    blind, berr = series("blind_separation")
    ekw = dict(ecolor=theme.INK, lw=0.7, capthick=0.7)
    ax.bar(x - w / 2, full, w, yerr=ferr, capsize=1.6, error_kw=ekw,
           color=[theme.DATASET[d] for d in DSS] + [theme.MUTED], zorder=3)
    ax.bar(x + w / 2, blind, w, yerr=berr, capsize=1.6, error_kw=ekw,
           color=theme.NEUTRAL, zorder=3)
    ax.axhline(0, color=theme.MUTED, lw=0.6)
    ax.axvline(2.5, color=theme.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("group separation $r$")
    ax.set_ylim(-0.135, 0.345)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    key = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    ax.legend([key, Patch(facecolor=theme.NEUTRAL)], ["persona", "blind"],
              handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=1.4,
              loc="upper left", ncol=2, columnspacing=0.8,
              labelspacing=0.2, handletextpad=0.3, borderpad=0.15, fontsize=5.5)

    fig.tight_layout(pad=0.25)
    p = os.path.join(FIGS, "pf_separation.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p




def fig_generated() -> str:
    """The generated-images result, in one strip.

    (left) the panel-size curve, (middle) per-pair panel preference against the
    human crowd. The cross-dataset calibration transfer that used to be the right
    panel now has its own appendix figure (``fig_calib_transfer``), since it is a
    property of the calibration and not of the generated-image task.
    """
    from collections import defaultdict
    from c3_rapidata import load_votes

    c3 = load("c3")

    # the per-pair arrays are not in c3.json (only their summaries are), so pool
    # the raw votes again here -- it is a dict walk, not a re-analysis
    pool_h, pool_p = defaultdict(list), defaultdict(list)
    for r in load_votes():
        pool_h[r["pair_id"]].append(r["human_choice"] == 2)   # image2 = flux
        pool_p[r["pair_id"]].append(r["pred_choice"] == 2)
    keys = list(pool_h)
    Hn = np.array([np.mean(pool_h[k]) for k in keys])
    Pn = np.array([np.mean(pool_p[k]) for k in keys])
    Nn = np.array([len(pool_h[k]) for k in keys])

    print_size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.17))

    # (left) panel-size curve
    nc = c3["aggregation"]["n_curve"]
    ns = sorted(int(k) for k in nc)
    ys = [nc[str(n)] for n in ns]
    maj = c3["aggregation"]["aggregate_acc_majority"]
    a1.axhline(maj, ls="--", lw=0.9, color=theme.REF, zorder=2, label="majority prior")
    a1.plot(ns, ys, "-o", ms=3, color=theme.PRIMARY, zorder=3, label="Rapidata")
    a1.annotate(f"{ys[0]:.3f}", (ns[0], ys[0]), textcoords="offset points",
                xytext=(3, 3), fontsize=5.5, color=theme.INK)
    a1.annotate(f"{ys[-1]:.3f}", (ns[-1], ys[-1]), textcoords="offset points",
                xytext=(-2, 4), fontsize=5.5, color=theme.INK, ha="right")
    a1.set_xscale("log")
    a1.set_xticks(ns)
    a1.set_xticklabels([str(n) for n in ns])
    a1.minorticks_off()
    a1.set_xlabel("panel size $N$")
    a1.set_ylabel("agreement\nw/ crowd")
    a1.set_ylim(min(min(ys), maj) - 0.016, max(ys) + 0.020)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    # the upper left is empty: the curve is still below the prior there
    a1.legend(loc="upper left", fontsize=5.5, handlelength=1.6, handletextpad=0.35,
              borderpad=0.2, labelspacing=0.2, framealpha=0.85)

    # (middle) per-pair scatter, points colored by how many humans voted on the
    # pair: rank terciles, so the three support groups stay balanced
    order = np.argsort(Nn, kind="stable")
    n = len(Nn)
    cat = np.empty(n, dtype=int)
    cat[order[: n // 3]] = 0
    cat[order[n // 3: 2 * n // 3]] = 1
    cat[order[2 * n // 3:]] = 2
    a2.plot([0, 1], [0, 1], ls="--", c=theme.REF, lw=0.8, zorder=1)
    a2.scatter(Hn, Pn, s=1.6, alpha=0.45, color=np.array(theme.BINS3)[cat],
               linewidth=0, zorder=2)
    a2.set_xlabel("human win-rate")
    a2.set_ylabel("panel win-rate")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    a2.set_xticks([0, 0.5, 1])
    a2.set_yticks([0, 0.5, 1])
    lo, hi = c3["aggregation"]["pair_corr_ci"]
    a2.text(0.04, 0.93, f"$r={c3['aggregation']['pair_corr']:+.3f}$\n95% CI [{lo:.3f}, {hi:.3f}]",
            transform=a2.transAxes, fontsize=5.0, va="top", color=theme.INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4)
    # the tercile colors need a key or they are an unexplained encoding; the
    # lower-right of this scatter is empty, so it costs nothing to put it there
    a2.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=2, color=c, label=l)
                       for c, l in zip(theme.BINS3, ["few", "med", "many"])],
              title="crowd votes", loc="lower right", fontsize=4.4,
              title_fontsize=4.4, handletextpad=0.15, labelspacing=0.12,
              borderpad=0.2, borderaxespad=0.25, framealpha=0.85)
    a2.grid(True)
    a2.set_axisbelow(True)

    fig.tight_layout(w_pad=1.0, pad=0.25)
    p = os.path.join(FIGS, "pf_generated.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight",
                pad_inches=0.02)  # thin margin keeps the caption close
    plt.close(fig)
    return p


def fig_calib_transfer(width: float = 3.2, height: float = 1.55) -> str:
    """Cross-dataset calibration transfer, for the calibration appendix.

    Rows: dataset the calibrator is evaluated on; columns: dataset it was fit on;
    cells: calibrated group MAE, with the raw (uncalibrated) error printed at the
    right as the reference every cell has to beat.
    """
    from matplotlib.colors import LinearSegmentedColormap

    xfer = load("calib_transfer")
    print_size()
    fig, a3 = plt.subplots(figsize=(width, height))
    # (right) calibration transfer. The printed numbers carry the values, so the
    # colorbar is dropped -- at this width it would cost more than it explains.
    cal = xfer["calibrated_group_mae_[eval][fitOn]"]
    raw = xfer["raw_group_mae"]
    M = np.array([[cal[ev][ft] for ft in DSS] for ev in DSS])
    cmap = LinearSegmentedColormap.from_list("seq", theme.SEQ)
    vmin, vmax = M.min(), M.max()
    a3.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    a3.grid(False)
    a3.set_xticks(range(3)); a3.set_xticklabels(DSS)
    a3.set_yticks(range(3)); a3.set_yticklabels(DSS)
    for i in range(3):
        for j in range(3):
            r, g, b, _ = cmap((M[i, j] - vmin) / (vmax - vmin + 1e-9))
            a3.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=5.2,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else theme.INK,
                    fontweight="bold" if i == j else "normal")
    # the uncalibrated error, as the reference every cell has to beat
    for i, ds in enumerate(DSS):
        a3.text(2.62, i, f"raw {raw[ds]:.2f}", ha="left", va="center",
                fontsize=5.0, color=theme.MUTED)
    a3.set_xlim(-0.5, 3.6)
    a3.set_xlabel("calibrator fit on")
    a3.set_ylabel("evaluated on")

    fig.tight_layout(pad=0.25)
    p = os.path.join(FIGS, "pf_calib_transfer.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_bias_wide() -> str:
    """Subgroup calibration gaps as vertical bars across the text width.

    The ``ax_bias`` version is a 21-row horizontal chart, which is nearly square
    and costs most of a page. Rotating it trades the long label column for a row
    of rotated ticks, so the same 21 bars fit in roughly half the height.
    """
    d = load("bias")
    rows = []
    for ds in DSS:
        for attr, m in d[ds]["by_attribute"].items():
            rows.append((ds, attr, m["bias_gap"]))
    rows.sort(key=lambda r: -r[2])          # largest first, so the eye starts on it

    def nice(attr: str) -> str:
        a = attr.replace("_", " ")
        a = a.replace("photographyExperience", "photo. exp.")
        a = a.replace("photographic level", "photo. level")
        a = a.replace("artExperience", "art exp.")
        a = a.replace("big5 ", "big5-")
        return a

    labels = [f"{ds}: {nice(at)}" for ds, at, _ in rows]
    vals = [v for _, _, v in rows]
    FAIR, MOD, SERIOUS_C = theme.SEV3       # light to dark navy

    def col(v):
        return SERIOUS_C if v > 0.15 else (MOD if v > 0.05 else FAIR)

    print_size()
    # the tick band is a fixed cost, so canvas height and printed height do not
    # move together; this keeps the plotting area at the height it had upright
    fig, ax = plt.subplots(figsize=(5.5, 1.451))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=[col(v) for v in vals], zorder=3, width=0.72)
    # the 0.05 line is the claim of the section: everything right of the third
    # bar sits under it, which is easier to see as a rule than as 18 numbers
    ax.axhline(0.05, ls=(0, (3, 1.6)), lw=0.8, color=theme.INK, zorder=4)
    # every bar is labelled, but the sub-threshold ones are lifted to a common
    # line just above the 0.05 rule: sitting on their own bar tops they would
    # run into the rule and into each other
    for xi, v in zip(x, vals):
        over = v > 0.05
        ax.text(xi, (v if over else 0.05) + 0.014, f"{v:.2f}",
                ha="center", va="bottom", fontsize=5.0,
                color=theme.INK if over else theme.MUTED)
    ax.set_xticks(x)
    # 45 deg rather than upright: the label band costs 0.585in instead of
    # 0.719in, and the canvas below gives back exactly that difference
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=4.8)
    ax.set_xlim(-0.8, len(labels) - 0.2)
    ax.set_ylim(0, max(vals) * 1.17)
    ax.set_ylabel("calibration gap")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=FAIR, label="fair ($\\leq$0.05)"),
                       Patch(facecolor=MOD, label="moderate"),
                       Patch(facecolor=SERIOUS_C, label="serious ($>$0.15)")],
              loc="upper right", fontsize=5.2, handlelength=1.4,
              handletextpad=0.35, labelspacing=0.22, borderpad=0.25)

    fig.tight_layout(pad=0.25)
    p = os.path.join(FIGS, "pf_bias.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_breadth_category() -> str:
    """Two checks on where the group prediction holds up.

    (left) every rated axis against the population-mean prior, (right) error by
    content category. Both were supplement-only; at this size they fit beside
    the robustness prose that already asserts the first of them.
    """
    dims = load("dims_extended")
    cats = load("content_category")["PARA"]["by_category"]

    print_size()
    # same height as ax_calibration_accv, so the appendix figures share one
    # short strip layout
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.21),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.12]))

    # (left) calibrated panel vs the prior, one point per rated axis. Below the
    # diagonal = the panel beats simply guessing the population mean.
    lim = max(max(r["pop_prior"] for r in dims),
              max(r["group_mae_cal"] for r in dims)) * 1.12
    a1.plot([0, lim], [0, lim], ls="--", c=theme.REF, lw=0.8, zorder=1)
    for ds in DSS:
        xs = [r["pop_prior"] for r in dims if r["dataset"] == ds]
        ys = [r["group_mae_cal"] for r in dims if r["dataset"] == ds]
        a1.scatter(xs, ys, s=9, alpha=0.9, color=theme.DATASET[ds],
                   edgecolor=theme.rim(theme.DATASET[ds]), linewidth=0.3, label=f"{ds} ({len(xs)})",
                   zorder=3)
    a1.set_xlim(0, lim); a1.set_ylim(0, lim)
    # same ticks on both axes, so the diagonal reads as equal error
    ticks = np.arange(0, lim, 0.04)
    a1.set_xticks(ticks); a1.set_yticks(ticks)
    a1.set_xlabel("population-mean prior")
    a1.set_ylabel("calibrated panel")
    a1.text(lim * 0.96, lim * 0.12, "below the line:\npanel wins", ha="right",
            va="bottom", fontsize=4.8, color=theme.MUTED, style="italic")
    a1.legend(loc="upper left", fontsize=4.8, handlelength=1.0,
              handletextpad=0.25, labelspacing=0.18, borderpad=0.2,
              borderaxespad=0.3)
    a1.grid(True); a1.set_axisbelow(True)

    # (right) difficulty by content category, hardest at the top
    items = sorted(cats.items(), key=lambda kv: kv[1]["group_mae"])
    # PARA's raw category keys, written out as words for the tick labels
    names = {"nightScene": "night scene", "stilllife": "still life"}
    labels = [names.get(k, k) for k, _ in items]
    vals = [v["group_mae"] for _, v in items]
    EASY, MOD, HARD = theme.SEV3
    t_easy, t_hard = 0.060, 0.075

    def col(v):
        return HARD if v >= t_hard else (MOD if v >= t_easy else EASY)

    y = np.arange(len(labels))
    a2.barh(y, vals, color=[col(v) for v in vals], zorder=3, height=0.72)
    a2.set_yticks(y); a2.set_yticklabels(labels, fontsize=4.8)
    for yi, v in zip(y, vals):
        a2.text(v + 0.0012, yi, f"{v:.3f}", va="center", fontsize=4.4,
                color=theme.INK)
    a2.set_xlim(0, max(vals) * 1.22)
    a2.set_xlabel("calibrated group MAE  (harder $\\rightarrow$)")
    a2.grid(True, axis="x"); a2.set_axisbelow(True)

    fig.tight_layout(w_pad=1.0, pad=0.25)
    p = os.path.join(FIGS, "pf_breadth_category.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p







# --------------------------------------------------------------------------
# Figure 3 — AutoPolish quantitative (2 panels)
# --------------------------------------------------------------------------
def fig_autopolish(logs_dir: str, drift_cap: float = 0.78, labels: dict | None = None,
                   out_name: str = "pf_autopolish.png", pad_inches: float = 0.1) -> str:
    labels = labels or C4LABELS   # per-venue legend names (ACCV: plain "reward-only")
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
        theme.rim_line(a1, steps, mean, theme.C4[c], plt.rcParams["lines.linewidth"], zorder=3)
        a1.plot(steps, mean, "-o", ms=2.5, color=theme.C4[c], label=labels[c], zorder=3,
                markeredgecolor=theme.rim(theme.C4[c], theme.C4[c]), markeredgewidth=0.4)
        a1.fill_between(steps, ci[:, 0], ci[:, 1], color=theme.C4[c], alpha=0.09, zorder=1)
    a1.set_xlabel("refinement step")
    a1.set_ylabel("best-so-far\nheld-out score")
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
        a2.scatter(xs, ys, s=4.5, alpha=0.72, color=theme.C4[c],
                   edgecolor=theme.rim(theme.C4[c], "none"), linewidth=0.3, zorder=3)
    a2.axvline(drift_cap, ls=(0, (3, 1.6)), lw=1.0, color=theme.INK, zorder=4)
    # headroom so the legend sits over empty plot rather than over the points
    ytop = a2.get_ylim()[1]
    a2.set_ylim(a2.get_ylim()[0], ytop * 1.20)
    a2.text(drift_cap + 0.008, ytop * 0.86, f"drift cap ({drift_cap:g})",
            va="top", ha="left", fontsize=5.5, color=theme.MUTED)
    a2.legend(handles=[Line2D([0], [0], marker="o", ms=2.4, lw=0,
                              color=theme.C4[c], markeredgewidth=0.3,
                              markeredgecolor=theme.rim(theme.C4[c], theme.C4[c]),
                              label=labels[c]) for c in present],
              loc="upper center", ncol=2, fontsize=5.5, borderpad=0.2,
              labelspacing=0.22, columnspacing=0.9, handlelength=1.0,
              handletextpad=0.3, framealpha=0.9)
    a2.set_xlabel("identity similarity of the committed best (DINOv2)")
    a2.set_ylabel("final held-out gain")
    a2.grid(True)
    a2.set_axisbelow(True)

    fig.tight_layout(w_pad=1.0, pad=0.25)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 4 — tight qualitative grid
# --------------------------------------------------------------------------
def _notes_column(fig, axes, row_ids, notes: dict, title: str, wrap: int | None = None,
                  label_pad: float = 1.6) -> None:
    """Fill the right-most axes column with a short text note per row, set off from the
    image grid by a dashed rule, with a short rule under the header that lines up with
    the top edge of the first row of photos. ``row_ids`` gives the image id of each row.
    Photo axes are anchored to the top of their cells, so positions are read after the
    aspect is applied."""
    import textwrap
    from matplotlib.lines import Line2D as _L
    for ax in axes[:, :-1].ravel():
        ax.apply_aspect()
    if wrap is None:
        col_in = axes[0, -1].get_position().width * fig.get_figwidth()
        wrap = max(12, int(col_in * 0.84 * 72 / (5.6 * 0.52)))   # ~0.52 em per glyph
    note = axes[0, -1].get_position()
    x_text = note.x0 + 0.14 * note.width
    for r, iid in enumerate(row_ids):
        axes[r, -1].axis("off")
        img = axes[r, 0].get_position()
        fig.text(x_text, (img.y0 + img.y1) / 2, textwrap.fill(notes.get(iid, ""), wrap),
                 ha="left", va="center", fontsize=5.6, color=theme.INK, linespacing=1.25)
    top = axes[0, 0].get_position().y1                    # top edge of the first photos
    bottom = min(axes[-1, c].get_position().y0 for c in range(axes.shape[1] - 1))
    head = top + label_pad / 72 / fig.get_figheight()       # header sits where the labels do
    fig.text(note.x0 + 0.5 * note.width + 0.04 * note.width, head, title, ha="center",
             va="bottom", fontsize=6, fontweight="bold", color=theme.INK)
    x = note.x0 + 0.06 * note.width
    fig.add_artist(_L([x, x], [bottom, head + 0.045], transform=fig.transFigure,
                      ls=(0, (3, 2)), lw=0.7, color="black"))
    fig.add_artist(_L([note.x0 + 0.20 * note.width, note.x1 - 0.12 * note.width], [top, top],
                      transform=fig.transFigure, lw=0.6, color="black"))

def fig_qualitative(logs_dir: str, edits_dir: str, n_show: int = 2, skip: int = 0,
                    row_offset: int = 1, out_name: str = "pf_qualitative.png",
                    last: str | None = None, notes: dict | None = None,
                    note_title: str = "Explanation", note_w: float = 1.25,
                    label_pad: float = 1.6, row_gap: float = 0.13, hspace: float = 0.10,
                    society_top: bool = False, reorder: bool = True,
                    pad_inches: float = 0.1) -> str:
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
        if society_top:   # only rows where AutoPolish has the (tied) best score
            best = max(finals[c].loc[iid]["best_obj"] for c in present)
            if finals["society"].loc[iid]["best_obj"] < best - 1e-9:
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
    picks = (apply_row_order(picks) if reorder else picks)[row_offset:row_offset + n_show]

    start = (data["society"][data["society"]["step"] == 0]
             .set_index("image_id")["best_obj"].to_dict())

    print_size()
    # ``last`` moves one condition to the right-most column (ACCV puts our method
    # last); the shared default keeps the CONDITIONS order for the other venues.
    order = [c for c in present if c != last] + ([last] if last in present else [])
    cols = ["source"] + order
    # Row height follows the images' own aspect ratio, so rows sit close
    # together instead of being separated by a band of unused axes.
    n_units = len(cols) + (note_w if notes else 0)
    cell_w = 5.5 / n_units
    aspects = []
    for iid in picks:
        with Image.open(_source_path(edits_dir, iid)) as im:
            aspects.append(im.height / im.width)
    # Each row is sized to its OWN aspect ratio. Using max(aspects) for every
    # row is fine while all rows are landscape, but one portrait row then
    # stretches the whole grid and leaves a dead band under every other row.
    row_hs = [cell_w * a + row_gap for a in aspects]   # + label line
    # Drawn at the final printed width so the per-cell labels stay legible.
    ncols = len(cols) + (1 if notes else 0)
    wr = [1.0] * len(cols) + ([note_w] if notes else [])
    fig, axes = plt.subplots(len(picks), ncols,
                             figsize=(5.5, sum(row_hs)),
                             gridspec_kw=dict(wspace=0.02, hspace=hspace,
                                              height_ratios=row_hs, width_ratios=wr))
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
                ax.set_anchor("N")
            # every row carries the full label: condition name and score. Our
            # method is marked with weight rather than hue, so the figure keeps
            # working in grayscale and for colorblind readers.
            label = f"{name}  {score:.2f}"
            weight = "bold" if col == "society" else "normal"
            ax.set_title(label, fontsize=6, pad=label_pad, fontweight=weight,
                         color=theme.INK)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    if notes:
        _notes_column(fig, axes, picks, notes, note_title, label_pad=label_pad)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 5 — the refinement loop over time (2 rows)
# --------------------------------------------------------------------------
# Which rows of the supplement's full 7-row progression grid to lift into the
# main text. Same ranking, so the main figure is a subset of the supplement's.
PROG_ROWS = (2, 6)


def fig_progression(logs_dir: str, edits_dir: str,
                    out_name: str = "pf_progression.png", notes: dict | None = None,
                    note_title: str = "Explanation", note_w: float = 1.25,
                    label_pad: float = 1.6, row_gap: float = 0.13, hspace: float = 0.16,
                    pad_inches: float = 0.1) -> str:
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
    cell_w = 5.5 / (len(CHECKPOINTS) + (note_w if notes else 0))
    aspects = []
    for _, _, _iid, cells in picks:
        with Image.open(cells[0][0]) as im:
            aspects.append(im.height / im.width)
    # Each row is sized to its OWN aspect ratio -- see fig_qualitative.
    row_hs = [cell_w * a + row_gap for a in aspects]   # + label line
    # drawn at the printed width so the per-cell labels stay legible
    ncols = len(CHECKPOINTS) + (1 if notes else 0)
    wr = [1.0] * len(CHECKPOINTS) + ([note_w] if notes else [])
    fig, axes = plt.subplots(len(picks), ncols,
                             figsize=(5.5, sum(row_hs)),
                             gridspec_kw=dict(wspace=0.02, hspace=hspace,
                                              height_ratios=row_hs, width_ratios=wr))
    axes = np.atleast_2d(axes)
    for r, (_, _, _iid, cells) in enumerate(picks):
        for c, (path, score, _) in enumerate(cells):
            ax = axes[r, c]
            ax.axis("off")
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_anchor("N")
            # every row carries the full label, as in fig_qualitative
            ax.set_title(f"{colnames[c]}  {score:.2f}", fontsize=6, pad=label_pad,
                         color=theme.INK)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    if notes:
        _notes_column(fig, axes, [iid for _, _, iid, _ in picks], notes, note_title,
                      label_pad=label_pad)
    p = os.path.join(FIGS, out_name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# ACCV example figures: three rows each, AutoPolish last, plus a note per row.
# The notes paraphrase the instruction AutoPolish actually committed (see the
# run logs) and what the edit visibly does; keep them in sync with the images.
# --------------------------------------------------------------------------
ACCV_PROG_ROWS = (2, 3, 6)
ACCV_PROG_NOTES = {
    "eva__698274": "The rabbit is made larger and sharper, so it stands out as the "
                   "subject.",
    "eva__696581": "The dark road and fields are brightened, so the foreground is no "
                   "longer lost.",
    "para__iaa_pub22564_.jpg": "More contrast and fur detail make the kitten stand out "
                               "slightly more.",
}
ACCV_QUAL_NOTES = {
    "para__iaa_pub3537_.jpg": "Plants and framed art make the bare office feel "
                              "lived-in.",
    "eva__696581": "A brighter, textured road adds detail below the sunset.",
    "eva__365027": "A player and a court turn a lone ball into a scene, which no other "
                   "critic tries.",
}


def fig_accv_examples(logs_dir: str, edits_dir: str) -> list:
    """Figures 7 and 8 of the ACCV draft (progression and critic comparison)."""
    global PROG_ROWS
    keep = PROG_ROWS
    PROG_ROWS = ACCV_PROG_ROWS
    try:
        prog = fig_progression(logs_dir, edits_dir, out_name="pf_progression_accv.png",
                               notes=ACCV_PROG_NOTES, label_pad=3.2, hspace=0.24,
                               pad_inches=0.02)  # thin margin keeps the caption close
    finally:
        PROG_ROWS = keep
    qual = fig_qualitative(logs_dir, edits_dir, n_show=3, out_name="pf_qualitative_accv.png",
                           last="society", notes=ACCV_QUAL_NOTES, label_pad=3.2, hspace=0.18,
                           pad_inches=0.02)
    return [prog, qual]


# Appendix A8/A9 grids in the same format (AutoPolish last, explanation column).
ACCV_APPX_PROG_ROWS = tuple(range(9))
ACCV_APPX_PROG_NOTES = {
    **ACCV_PROG_NOTES,
    "para__iaa_pub18208_.jpg": "Colors become slightly more vivid, a small change with a "
                               "small gain.",
    "eva__489134": "A Milky Way and bright stars appear over the road, turning the plain "
                   "night sky into the subject.",
    "para__iaa_pub10458_.jpg": "The food is brightened and its colors made richer, so the "
                               "plate looks fresher.",
    "eva__55012": "The sky is made brighter and bluer, so the dark structure stands out "
                  "more against it.",
    "eva__412312": "Haze is cleared from the scene, so the lights and traffic look "
                   "sharper.",
    "para__iaa_pub9304_.jpg": "The crystals are lit more evenly and brightened, so their "
                              "detail is easier to see.",
}
ACCV_APPX_QUAL_NOTES = {
    **ACCV_QUAL_NOTES,
    "eva__18116": "The green bench is turned bright red, giving the gray wall a clear "
                  "point of focus.",
    "eva__489134": "A bright Milky Way and many stars fill the empty sky, making it the "
                   "subject of the photo.",
    "eva__742534": "The steel ovens behind the chef become a bright kitchen, giving the "
                   "photo a warmer setting.",
    "eva__55012": "The sky is made brighter and bluer, so the dark structure stands out "
                  "against it.",
    "eva__444303": "A truck is added to the empty snowy courtyard, giving the scene scale "
                   "and a point of interest.",
    "eva__449": "The black night sky becomes a pale background, so the tower's structure "
                "is easier to see.",
    "eva__126207": "Keeps the black-and-white look but deepens the contrast between the "
                   "building and the sky.",
    "para__iaa_pub29955_.jpg": "The sky gets a deeper blue and a brighter band of sunset "
                               "color along the horizon.",
    "eva__802648": "The busy green background becomes plain gray, so attention stays on "
                   "the face.",
    "para__iaa_pub21242_.jpg": "Softer light and slightly richer colors make the food look "
                               "a little more appetizing.",
}


def _fit_aspect(render, target: float, lo: float = 0.02, hi: float = 0.9) -> str:
    """Binary-search the per-row gap (inches) so the saved PNG's height/width matches
    ``target``: the figure then fills a full LNCS page next to its caption. Only the
    whitespace between rows changes; the photos keep their size."""
    def aspect(gap):
        path = render(gap)
        with Image.open(path) as im:
            return path, im.height / im.width
    path, a = aspect(lo)
    if a >= target:
        return path
    for _ in range(14):
        mid = (lo + hi) / 2
        path, a = aspect(mid)
        if abs(a - target) < 0.002:
            break
        lo, hi = (mid, hi) if a < target else (lo, mid)
    return path


# Full page in llncs: 12.2 x 19.3 cm, minus a two-line caption and its skip.
ACCV_FULLPAGE_ASPECT = (19.3 - 1.28) / 12.2
# Per-figure targets: each figure gives back the extra height its caption takes
# beyond two lines (measured from LaTeX's "Float too large" overflow, plus ~2.5pt of
# slack). Re-measure if a caption is rewritten.
_PT = 2.54 / 72.27
ACCV_MORE_ASPECT = (19.3 - 1.28 - (17.705 + 2.5) * _PT) / 12.2   # A6, caption ~3 lines
ACCV_PROG_ASPECT = (19.3 - 1.28 - (6.857 + 2.5) * _PT) / 12.2    # A7, caption 3 lines


def fig_accv_appendix_examples(logs_dir: str, edits_dir: str) -> list:
    """Appendix grids: the top-five comparison, further rows where AutoPolish scores
    best, and the loop over time. The last two are sized to fill a page exactly."""
    global PROG_ROWS
    kw = dict(label_pad=3.2, hspace=0.18, last="society")
    top5 = fig_qualitative(logs_dir, edits_dir, n_show=5, row_offset=0,
                           out_name="ax_qualitative_top5_accv.png",
                           notes=ACCV_APPX_QUAL_NOTES, **kw)
    more = _fit_aspect(lambda gap: fig_qualitative(
        logs_dir, edits_dir, n_show=8, skip=5, row_offset=0, society_top=True,
        reorder=False, out_name="ax_qualitative_more_accv.png",
        notes=ACCV_APPX_QUAL_NOTES, row_gap=gap, **kw), ACCV_MORE_ASPECT)
    keep = PROG_ROWS
    PROG_ROWS = ACCV_APPX_PROG_ROWS
    try:
        prog = _fit_aspect(lambda gap: fig_progression(
            logs_dir, edits_dir, out_name="ax_progression_accv.png",
            notes=ACCV_APPX_PROG_NOTES, label_pad=3.2, hspace=0.24, row_gap=gap),
            ACCV_PROG_ASPECT)
    finally:
        PROG_ROWS = keep
    return [top5, more, prog]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Composite main-text paper figures.")
    ap.add_argument("--c4-root", default=os.path.join(REPO, "data", "results", "c4_run2"))
    ap.add_argument("--n-show", type=int, default=2)
    ap.add_argument("--figs", default=FIGS, help="output figure dir (one per venue)")
    ap.add_argument("--prog-rows", default=None,
                    help="comma-separated row indices for the progression figure "
                         "(default: %s)" % ",".join(str(r) for r in PROG_ROWS))
    ap.add_argument("--dpi", type=int, default=DPI,
                    help="raster DPI for the saved figures (default %(default)s)")
    ap.add_argument("--suffix", default="",
                    help="appended to the qualitative/progression filenames, so a "
                         "variant can live beside the originals in a shared figs dir")
    args = ap.parse_args()
    DPI = args.dpi
    if args.prog_rows:
        PROG_ROWS = tuple(int(x) for x in args.prog_rows.split(","))
    FIGS = args.figs
    logs = os.path.join(args.c4_root, "logs")
    edits = os.path.join(args.c4_root, "edits")

    os.makedirs(FIGS, exist_ok=True)
    print("wrote", fig_audience())
    print("wrote", fig_persona())
    print("wrote", fig_steer())
    print("wrote", fig_separation())
    print("wrote", fig_generated())
    print("wrote", fig_calib_transfer())
    print("wrote", fig_bias_wide())
    print("wrote", fig_breadth_category())
    print("wrote", fig_autopolish(logs))
    print("wrote", fig_autopolish(logs, labels={**C4LABELS, "reward_only": "reward-only"},
                                  out_name="pf_autopolish_accv.png", pad_inches=0.02))
    print("wrote", fig_qualitative(logs, edits, n_show=args.n_show,
                                   out_name="pf_qualitative%s.png" % args.suffix))
    print("wrote", fig_accv_examples(logs, edits))
    print("wrote", fig_progression(logs, edits,
                                   out_name="pf_progression%s.png" % args.suffix))
