"""Shared visual theme for all paper figures.

One palette, one look, applied across every plot so the paper reads as a single
visual system. The categorical hues are navy, magenta and yellow from the
Okabe-Ito set (all pairs CVD-safe); gray is a reserved neutral for baselines,
never a "real" category.

Yellow sits at 1.3:1 against white, so it works as a bar fill but a yellow
line or dot needs ``YELLOW_RIM`` around it to be seen; ``rim`` and ``rim_line``
supply that.

Usage in a plotting script::

    import theme; theme.apply()
    ax.bar(..., color=theme.PRIMARY)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# --- categorical palette (validated: slots 1-3 pass all-pairs CVD) ---
NAVY = "#0072B2"     # primary   — "ours": society / persona / panel
MAGENTA = "#CC79A7"  # secondary — control: blind / no-persona
YELLOW = "#F0E442"   # tertiary  — oracle: reward-only
GRAY = "#BDBDBD"     # neutral   — baseline fills; a darker gray reads as
                     # magenta under deuteranopia

# semantic aliases
PRIMARY, SECONDARY, TERTIARY, NEUTRAL = NAVY, MAGENTA, YELLOW, GRAY

# dashed reference lines (diagonals, priors): darker than GRAY so a thin dash
# still shows, and a line is never mistaken for a magenta fill
REF = "#8A8A86"

# dark olive rim that keeps yellow lines and dots visible on white (4:1)
YELLOW_RIM = "#8C8200"

# ink + surface
INK = "#1a1a19"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#ffffff"

# raw-vs-calibrated encoding (calibration figure)
RAW = YELLOW
CAL = NAVY
PRIOR = MAGENTA

# per-dataset color, shared across the steerability and C1 figures
DATASET = {"PARA": MAGENTA, "EVA": NAVY, "LAPIS": YELLOW}

# ordered levels take one hue, light -> dark, so the order reads in the color;
# the light end still clears 2:1 on white
NAVY3 = ["#7FB2D6", NAVY, "#003D61"]

# ordered 3-bin support (e.g. few / medium / many votes)
BINS3 = NAVY3

# low -> mid -> high severity/score spectrum
SEV3 = NAVY3

# sequential colormap stops (heatmaps): near-white -> navy
SEQ = ["#EEF5FA", NAVY]

# C4 condition -> color (color follows the entity, not its rank)
C4 = {"static": GRAY, "blind": MAGENTA, "society": NAVY, "reward_only": YELLOW}


def rim(color, default: str = "white") -> str:
    """Edge color for a dot or marker of ``color``: yellow gets the dark rim,
    every other hue keeps ``default``."""
    return YELLOW_RIM if color == YELLOW else default


def rim_line(ax, x, y, color, lw: float, zorder: float) -> None:
    """Draw a slightly wider dark line under a yellow line so it reads on
    white; other hues need none. An underlay rather than a path effect, which
    would also stroke the markers and draw them larger than the other series'."""
    if color == YELLOW:
        ax.plot(x, y, "-", color=YELLOW_RIM, lw=lw + 0.5, zorder=zorder - 0.01)


def ink_on(color) -> str:
    """Hatch or text color that reads on top of a fill of ``color``."""
    r, g, b = to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else MUTED


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
