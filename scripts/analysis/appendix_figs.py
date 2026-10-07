"""Appendix figures for the ACCV build, drawn in the style of the main text.

The supplement used to carry standalone per-experiment plots (b4_calibration,
ax_response_style) drawn at a larger size with bold panel titles. This script
redraws their numbers as one wide, low, three-panel figure with the same type
sizes, encoding and in-axes legends as ``pf_persona``, so the appendix reads
like the main text.

  * ax_calibration_accv.png — (left) share of answers on the most common value
    (middle) rating entropy  (right) error of a single rating and of a group
    average, before and after calibration

Pure re-analysis (no GPU). Run from ``scripts/analysis/``::

    python appendix_figs.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

import theme
from paper_figs import DPI, DSS, FIGS, load, print_size

HATCH = "//////"


def _bars_vs(ax, model, human, ylabel, ylim):
    """VLM bars in the dataset hue beside gray human bars, as in pf_persona."""
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, model, w, color=[theme.DATASET[d] for d in DSS], zorder=3)
    ax.bar(x + w / 2, human, w, color=theme.NEUTRAL, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(DSS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    key = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    ax.legend([key, Patch(facecolor=theme.NEUTRAL)], ["VLM", "human"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper center", ncol=2, columnspacing=0.7, handlelength=1.6,
              handletextpad=0.35, borderpad=0.1, fontsize=5.5)


def fig_calibration() -> str:
    """Why the audience's ratings need calibration, and what it fixes."""
    style = load("response_style")
    cal = load("calibration")

    print_size()
    plt.rcParams["hatch.linewidth"] = 0.35
    plt.rcParams["hatch.color"] = "white"
    # widths follow the bars per group: 2 (VLM, human), 2, and 4 (raw/calibrated x single/group)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(5.5, 1.21),
                                     gridspec_kw={"width_ratios": [2, 2, 4]})

    # (left) the VLM piles its answers onto one value far more than people do
    vm = [style[d]["vlm"]["modal_share"] for d in DSS]
    hm = [style[d]["human"]["modal_share"] for d in DSS]
    _bars_vs(a1, vm, hm, "share of answers on\nthe most common value", (0, 0.8))

    # (middle) and so spreads them over fewer values on the wider scales
    ve = [style[d]["vlm"]["entropy_bits"] for d in DSS]
    he = [style[d]["human"]["entropy_bits"] for d in DSS]
    _bars_vs(a2, ve, he, "rating entropy (bits)", (0, 6.0))

    # (right) calibration helps the group average far more than one rating.
    # Raw bars carry the dataset hue and calibrated ones are gray, as in the
    # calibration panel of pf_persona; hatching marks the single rating. The
    # last group is the mean of the three datasets (all on the same normalized
    # scale), drawn in dark gray and set off by a dashed rule as in pf_separation.
    labels = DSS + ["Average"]
    x = np.arange(len(labels))
    w = 0.2
    hue = [theme.DATASET[d] for d in DSS] + [theme.MUTED]

    def with_mean(stage, key):
        vals = [cal[d][stage][key] for d in DSS]
        return vals + [float(np.mean(vals))]

    series = [
        (with_mean("raw", "individual_mae"), hue, HATCH),
        (with_mean("calibrated", "individual_mae"), theme.NEUTRAL, HATCH),
        (with_mean("raw", "group_mae"), hue, None),
        (with_mean("calibrated", "group_mae"), theme.NEUTRAL, None),
    ]
    for k, (vals, color, hatch) in enumerate(series):
        bars = a3.bar(x + (k - 1.5) * w, vals, w, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0, zorder=3)
        # white hatching vanishes on the light fills (yellow, gray)
        for b in bars:
            b.set_hatchcolor(theme.ink_on(b.get_facecolor()))
    a3.axvline(2.5, color=theme.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    a3.set_xticks(x)
    a3.set_xticklabels(labels)
    a3.set_ylabel("rating error (MAE)")
    a3.set_ylim(0, 0.42)
    a3.grid(True, axis="y")
    a3.set_axisbelow(True)
    key = tuple(Patch(facecolor=theme.DATASET[d]) for d in DSS)
    a3.legend([key, Patch(facecolor=theme.NEUTRAL),
               Patch(facecolor=theme.MUTED, hatch=HATCH, edgecolor="white", linewidth=0),
               Patch(facecolor=theme.MUTED)],
              ["raw", "calib.", "single", "group"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper center", ncol=2, columnspacing=0.7, handlelength=1.6,
              handletextpad=0.35, borderpad=0.1, labelspacing=0.2, fontsize=5.5)

    fig.tight_layout(w_pad=0.9, pad=0.25)
    p = os.path.join(FIGS, "ax_calibration_accv.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", fig_calibration())
