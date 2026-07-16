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

from common import OUT_DIR, REPO, ensure_out

C4_RESULTS = os.path.join(REPO, "data", "results")
CONDITIONS = ["static", "blind", "society", "reward_only"]
LABELS = {"static": "static", "blind": "blind VLM", "society": "society",
          "reward_only": "reward-only"}
N_SHOW = 5


def load_c4(condition: str) -> pd.DataFrame:
    run = f"c4_{condition}"
    parts = sorted(glob.glob(os.path.join(C4_RESULTS, run, f"{run}*.part-*.json")))
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
    ensure_out()
    figs = os.path.join(OUT_DIR, "figs")
    os.makedirs(figs, exist_ok=True)

    data = {c: load_c4(c) for c in CONDITIONS}
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

    cols = ["source"] + present
    fig, axes = plt.subplots(len(picks), len(cols),
                             figsize=(2.6 * len(cols), 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, img_id in enumerate(picks):
        src = finals["society"].loc[img_id]
        # source lives next to any condition's step0 file
        cond_dir = os.path.join(REPO, "data", "c4_edits", "society", img_id)
        src_path = os.path.join(cond_dir, "step0_source.png")
        for cc, col in enumerate(cols):
            ax = axes[r, cc]
            ax.axis("off")
            if col == "source":
                path = src_path
                title = f"source\n(aes {finals['society'].loc[img_id]['best_obj']:.2f} start)"
                # step0 best_obj is the source score; use step0 row if available
            else:
                path = str(finals[col].loc[img_id]["best_path"]) if img_id in finals[col].index else None
                score = finals[col].loc[img_id]["best_obj"] if img_id in finals[col].index else float("nan")
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
