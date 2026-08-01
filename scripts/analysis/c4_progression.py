"""C4 progression figure — the AutoPolish loop over time.

For the society run, lays out (rows = images) the best-so-far image at a fixed set
of refinement steps (columns), so each row reads left-to-right as the image being
polished. Picks images whose best keeps improving across steps (most visible
change) and that are fully renderable from the local edits. Run from
``scripts/analysis/``.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import theme
from c4_qualitative import load_c4, _source_path

CHECKPOINTS = [0, 2, 4, 6, 10]


def _cell(g, step, edits_dir, iid):
    """(path, score, basename) for the best-so-far at `step`."""
    row = g[g["step"] <= step].sort_values("step").iloc[-1]
    bp = os.path.basename(str(row["best_path"]))
    if step == 0 or "source" in bp:
        path = _source_path(edits_dir, iid)
    else:
        path = os.path.join(edits_dir, "society", iid, bp)
    return path, float(row["best_obj"]), bp


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 refinement-over-time grid.")
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--edits-dir", required=True)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--n-show", type=int, default=7)
    ap.add_argument("--out-name", default="ax_progression.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    figs = os.path.join(args.analysis_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    df = load_c4("society", args.logs_dir)
    scored = []
    for iid, g in df.groupby("image_id"):
        g = g.sort_values("step")
        cells = [_cell(g, s, args.edits_dir, iid) for s in CHECKPOINTS]
        if any(p is None or not os.path.exists(p) for p, _, _ in cells):
            continue
        distinct = len({os.path.basename(p) for p, _, _ in cells})
        gain = cells[-1][1] - cells[0][1]
        scored.append((distinct, gain, iid, cells))
    # most visible progression first (distinct checkpoints), then largest gain
    scored.sort(key=lambda r: (-r[0], -r[1]))
    picks = scored[:args.n_show]

    theme.apply()
    colnames = ["source"] + [f"step {s}" for s in CHECKPOINTS[1:]]
    ncol = len(CHECKPOINTS)
    fig, axes = plt.subplots(len(picks), ncol,
                             figsize=(2.6 * ncol, 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (_, _, iid, cells) in enumerate(picks):
        for c, (path, score, _) in enumerate(cells):
            ax = axes[r, c]
            ax.axis("off")
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(f"{colnames[c]}\n(aes {score:.2f})", fontsize=8)
    if args.title:
        fig.suptitle(args.title, y=1.0, fontsize=11)
    fig.tight_layout()
    out = os.path.join(figs, args.out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(picks)} images, cols={CHECKPOINTS})")


if __name__ == "__main__":
    main()
