"""C4 qualitative figure — original vs best-edit-per-condition grid.

Picks the images where the society critic helped most (largest society-minus-
static final-best gain) and lays out, per image, the source next to each
condition's committed best image. Reads the PNGs saved by ``script/c4_refine.py``
under ``data/c4_edits/``. Run from ``scripts/analysis/``.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import argparse

from common import REPO

DEFAULT_OUTPUT_ROOT = os.path.join(REPO, "outputs", "c4_auto_research")
CONDITIONS = ["static", "blind", "society", "reward_only"]
LABELS = {"static": "static", "blind": "blind VLM", "society": "society",
          "reward_only": "reward-only"}
N_SHOW = 5


def load_c4(condition: str, logs_dir: str) -> pd.DataFrame:
    run = f"c4_{condition}"
    parts = sorted(glob.glob(os.path.join(logs_dir, run, f"{run}*.part-*.json")))
    rows: list[dict] = []
    for p in parts:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return pd.DataFrame(rows)


def _final_best(df: pd.DataFrame) -> pd.DataFrame:
    """Last-step row per image (carries best_obj + best_path)."""
    idx = df.groupby("image_id")["step"].idxmax()
    return df.loc[idx].set_index("image_id")


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 qualitative before/after grid.")
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                    help="same root passed to script/c4_refine.py (default: %(default)s).")
    ap.add_argument("--logs-dir", default=None, help="override <root>/logs.")
    ap.add_argument("--edits-dir", default=None, help="override <root>/edits.")
    ap.add_argument("--analysis-dir", default=None, help="override <root>/analysis.")
    args = ap.parse_args()
    logs_dir = args.logs_dir or os.path.join(args.output_root, "logs")
    edits_dir = args.edits_dir or os.path.join(args.output_root, "edits")
    figs = os.path.join(args.analysis_dir or os.path.join(args.output_root, "analysis"), "figs")
    os.makedirs(figs, exist_ok=True)

    data = {c: load_c4(c, logs_dir) for c in CONDITIONS}
    present = [c for c in CONDITIONS if len(data[c])]
    finals = {c: _final_best(data[c]) for c in present}
    if "society" not in present:
        raise SystemExit("need the society run for the qualitative figure.")

    # rank images by society-minus-static final best_obj gain
    soc = finals["society"]["best_obj"]
    base = finals["static"]["best_obj"] if "static" in present else soc * 0
    common = soc.index.intersection(base.index) if "static" in present else soc.index
    gain = (soc.loc[common] - base.loc[common]).sort_values(ascending=False)
    picks = list(gain.index[:N_SHOW])

    # true source aesthetic = step-0 best_obj (identical across conditions)
    start_score = (data["society"][data["society"]["step"] == 0]
                   .set_index("image_id")["best_obj"].to_dict())

    cols = ["source"] + present
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(2.6 * len(cols), 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, img_id in enumerate(picks):
        # source lives next to the society run's step0 file
        src_path = os.path.join(edits_dir, "society", img_id, "step0_source.png")
        for cc, col in enumerate(cols):
            ax = axes[r, cc]
            ax.axis("off")
            if col == "source":
                path = src_path
                title = f"source\n(aes {start_score.get(img_id, float('nan')):.2f})"
            else:
                # Reconstruct from the LOCAL edits_dir (best_path in the log is an
                # absolute path from wherever the run executed, e.g. Colab/Drive).
                if img_id in finals[col].index:
                    bp = str(finals[col].loc[img_id]["best_path"])
                    path = os.path.join(edits_dir, col, img_id, os.path.basename(bp))
                    score = finals[col].loc[img_id]["best_obj"]
                else:
                    path, score = None, float("nan")
                title = f"{LABELS[col]}\n(aes {score:.2f})"
            if path and os.path.exists(path):
                ax.imshow(Image.open(path).convert("RGB"))
            if r == 0 or col == "source":
                ax.set_title(title, fontsize=8)
            else:
                ax.set_title(title, fontsize=8)
    fig.suptitle("C4 — original vs best edit per condition (top society gains)", y=1.0)
    fig.tight_layout()
    out = os.path.join(figs, "c4_qualitative.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(picks)} images)")


if __name__ == "__main__":
    main()
