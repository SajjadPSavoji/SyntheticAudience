"""Extra publishable figures for the paper, built from the cached result JSONs.

Pure re-analysis (no GPU): reads results/*.json and writes themed PNGs into
results/figs/. All styling comes from theme.py so these match the main figures.

Figures produced:
  * ax_calib_transfer.png  — calibration transfers across datasets (heatmap)
  * ax_bias.png            — subgroup calibration fairness (severity bars)
  * ax_response_style.png  — the judge's central-tendency bias (human vs VLM)
  * ax_breadth.png         — group beats the population prior on all 16 axes
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import theme

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
FIGS = os.path.join(RES, "figs")
DSS = ["PARA", "EVA", "LAPIS"]

# severity colors for the fairness figure (status palette, kept apart from series hues)
SERIOUS = "#C0392B"


def load(name):
    return json.load(open(os.path.join(RES, f"{name}.json"), encoding="utf-8"))


def fig_calib_transfer():
    d = load("calib_transfer")
    raw = d["raw_group_mae"]
    cal = d["calibrated_group_mae_[eval][fitOn]"]
    M = np.array([[cal[ev][ft] for ft in DSS] for ev in DSS])  # rows=eval, cols=fitOn
    theme.apply()
    # green (best) -> orange (worst) spectrum across the actual cells
    cmap = LinearSegmentedColormap.from_list("go", [theme.AQUA, theme.ORANGE])
    vmin, vmax = M.min(), M.max()
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"fit on\n{d}" for d in DSS])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"eval {d}" for d in DSS])
    for i in range(3):
        for j in range(3):
            val = M[i, j]
            r, g, b, _ = cmap((val - vmin) / (vmax - vmin + 1e-9))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="white" if lum < 0.6 else theme.INK,
                    fontweight="bold" if i == j else "normal", fontsize=11)
    # raw (uncalibrated) reference column on the right
    for i, ds in enumerate(DSS):
        ax.text(3.05, i, f"raw\n{raw[ds]:.3f}", ha="left", va="center",
                color=SERIOUS, fontsize=9)
    ax.set_xlim(-0.5, 3.0)
    ax.set_xlabel("calibrator fit on")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.16)
    cbar.set_label("group MAE (lower is better)")
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_calib_transfer.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_bias():
    d = load("bias")
    rows = []
    for ds in DSS:
        for attr, m in d[ds]["by_attribute"].items():
            rows.append((f"{ds}: {attr}", m["bias_gap"]))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    FAIR, MOD, SERIOUS_C = theme.SEV3  # green, blue, orange

    def col(v):
        if v > 0.15:
            return SERIOUS_C
        if v > 0.05:
            return MOD
        return FAIR

    colors = [col(v) for v in vals]
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    xmax = max(vals) * 1.15
    ax.set_xlim(0, xmax)
    ax.text(0.05 / xmax, 1.012, "fair ($<$0.05)", transform=ax.transAxes,
            color=theme.MUTED, fontsize=9, ha="center", va="bottom")
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=8, color=theme.INK)
    ax.set_xlabel("subgroup calibration gap  (max$-$min bias across levels)")
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    handles = [Patch(facecolor=FAIR, label="fair ($\\leq$0.05)"),
               Patch(facecolor=MOD, label="moderate"),
               Patch(facecolor=SERIOUS_C, label="serious ($>$0.15)")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_bias.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def _dataset_legend(control_label):
    """Shared key: colored bars are the model, tinted by dataset; gray is the control."""
    return ([Patch(facecolor=theme.DATASET[d], label=d) for d in DSS]
            + [Patch(facecolor=theme.NEUTRAL, label=control_label)])


def fig_response_style():
    d = load("response_style")
    theme.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.1))
    x = np.arange(3); w = 0.38
    vlm_c = [theme.DATASET[ds] for ds in DSS]
    hm = [d[ds]["human"]["modal_share"] for ds in DSS]
    vm = [d[ds]["vlm"]["modal_share"] for ds in DSS]
    a1.bar(x - w/2, hm, w, color=theme.NEUTRAL, zorder=3)
    a1.bar(x + w/2, vm, w, color=vlm_c, zorder=3)
    a1.set_xticks(x); a1.set_xticklabels(DSS)
    a1.set_ylabel("share of answers on the single\nmost common value")
    a1.set_title("Central-tendency bias", fontsize=11)
    a1.grid(True, axis="y"); a1.set_axisbelow(True)

    he = [d[ds]["human"]["entropy_bits"] for ds in DSS]
    ve = [d[ds]["vlm"]["entropy_bits"] for ds in DSS]
    a2.bar(x - w/2, he, w, color=theme.NEUTRAL, zorder=3)
    a2.bar(x + w/2, ve, w, color=vlm_c, zorder=3)
    a2.set_xticks(x); a2.set_xticklabels(DSS)
    a2.set_ylabel("rating entropy (bits)")
    a2.set_title("Answer diversity", fontsize=11)
    a2.grid(True, axis="y"); a2.set_axisbelow(True)

    fig.legend(handles=_dataset_legend("human (both panels)"),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_response_style.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_breadth():
    d = load("dims_extended")
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.4, 5))
    lim = 0.0
    for r in d:
        lim = max(lim, r["pop_prior"], r["group_mae_cal"])
    lim *= 1.1
    ax.plot([0, lim], [0, lim], ls="--", c=theme.NEUTRAL, lw=1.0, zorder=1)
    for ds in DSS:
        xs = [r["pop_prior"] for r in d if r["dataset"] == ds]
        ys = [r["group_mae_cal"] for r in d if r["dataset"] == ds]
        unit = "axis" if len(xs) == 1 else "axes"
        ax.scatter(xs, ys, s=55, alpha=0.85, color=theme.DATASET[ds],
                   edgecolor="white", linewidth=0.6, label=f"{ds} ({len(xs)} {unit})", zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("population-mean prior  (group MAE)")
    ax.set_ylabel("calibrated panel  (group MAE)")
    ax.text(lim * 0.97, lim * 0.9, "below the line:\npanel beats the prior",
            ha="right", va="top", fontsize=9, color=theme.MUTED, style="italic")
    ax.legend(loc="upper left"); ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_breadth.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_rationale():
    d = load("rationale")
    theme.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.1))
    x = np.arange(3); w = 0.38
    pc = [theme.DATASET[ds] for ds in DSS]  # persona bars, per dataset
    fa = [d[ds]["auc_full"] for ds in DSS]
    ba = [d[ds]["auc_blind"] for ds in DSS]
    a1.axhline(0.5, ls="--", lw=1.4, color=theme.INK, zorder=4)
    a1.bar(x - w/2, fa, w, color=pc, zorder=3)
    a1.bar(x + w/2, ba, w, color=theme.NEUTRAL, zorder=3)
    a1.set_xticks(x); a1.set_xticklabels(DSS); a1.set_ylim(0.45, 0.75)
    a1.set_ylabel("rater attribute recoverable\nfrom rationale text (AUC)")
    a1.set_title("Persona steers the language", fontsize=11)
    a1.grid(True, axis="y"); a1.set_axisbelow(True)

    fd = [d[ds]["rationale_diversity_full"] for ds in DSS]
    bd = [d[ds]["rationale_diversity_blind"] for ds in DSS]
    a2.bar(x - w/2, fd, w, color=pc, zorder=3)
    a2.bar(x + w/2, bd, w, color=theme.NEUTRAL, zorder=3)
    a2.set_xticks(x); a2.set_xticklabels(DSS)
    a2.set_ylabel("distinct-rationale fraction")
    a2.set_title("Persona diversifies the rationales", fontsize=11)
    a2.grid(True, axis="y"); a2.set_axisbelow(True)

    handles = _dataset_legend("no-persona") + [
        Line2D([0], [0], ls="--", lw=1.4, color=theme.INK, label="chance (0.5)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_rationale.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_content_category():
    d = load("content_category")["PARA"]["by_category"]
    items = sorted(d.items(), key=lambda kv: kv[1]["group_mae"])
    labels = [k for k, _ in items]
    vals = [v["group_mae"] for _, v in items]
    theme.apply()
    EASY, MOD, HARD = theme.SEV3  # green, blue, orange
    t_easy, t_hard = 0.060, 0.075  # thresholds on calibrated group MAE

    def col(v):
        return HARD if v >= t_hard else (MOD if v >= t_easy else EASY)

    colors = [col(v) for v in vals]
    hi = max(vals)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0012, yi, f"{v:.3f}", va="center", fontsize=8, color=theme.INK)
    ax.set_xlabel("calibrated group MAE  (harder $\\rightarrow$)")
    ax.set_xlim(0, hi * 1.18)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    handles = [Patch(facecolor=EASY, label="easy ($<$0.060)"),
               Patch(facecolor=MOD, label="moderate"),
               Patch(facecolor=HARD, label="hard ($\\geq$0.075)")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIGS, "ax_content_category.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    for f in (fig_calib_transfer, fig_bias, fig_response_style, fig_breadth,
              fig_rationale, fig_content_category):
        print("wrote", f())
