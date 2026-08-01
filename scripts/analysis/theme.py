"""Shared visual theme for all paper figures.

One palette, one look, applied across every plot so the paper reads as a single
visual system. The categorical hues are the first three slots of a
colorblind-validated palette (blue/orange/aqua all-pairs CVD-safe); gray is a
reserved neutral for baselines and reference lines, never a "real" category.

Usage in a plotting script::

    import theme; theme.apply()
    ax.bar(..., color=theme.PRIMARY)
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# --- categorical palette (validated: slots 1-3 pass all-pairs CVD) ---
BLUE = "#2a78d6"    # primary   — "ours": society / persona / panel
ORANGE = "#eb6834"  # secondary — control: blind / no-persona
AQUA = "#1baf7a"    # tertiary  — oracle: reward-only
GRAY = "#8a8a86"    # neutral   — baseline / priors / reference lines

# semantic aliases
PRIMARY, SECONDARY, TERTIARY, NEUTRAL = BLUE, ORANGE, AQUA, GRAY

# ink + surface
INK = "#1a1a19"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#ffffff"

# raw-vs-calibrated encoding (calibration figure)
RAW = AQUA      # green
CAL = BLUE
PRIOR = ORANGE

# per-dataset color, shared across the steerability and C1 figures
DATASET = {"PARA": ORANGE, "EVA": BLUE, "LAPIS": AQUA}

# ordered 3-bin categorical (e.g. low/medium/high support)
BINS3 = [ORANGE, BLUE, AQUA]

# low -> mid -> high severity/score spectrum (green -> blue -> orange)
SEV3 = [AQUA, BLUE, ORANGE]

# C4 condition -> color (color follows the entity, not its rank)
C4 = {"static": GRAY, "blind": ORANGE, "society": BLUE, "reward_only": AQUA}


def apply() -> None:
    """Install the theme into matplotlib rcParams (idempotent)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5,
        "axes.labelcolor": MUTED,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0,
    })
