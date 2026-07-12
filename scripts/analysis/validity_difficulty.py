"""#2 Difficulty as a validity probe (EVA). Pure re-analysis.

EVA raters give a 'difficulty' rating (how hard the image was to judge). If the VLM's predicted
difficulty is meaningful, it should track where humans actually DISAGREE (the per-image spread of
the human aesthetic score). Three correlations, per image:
  - human difficulty  vs human score disagreement  -> is difficulty even a valid construct?
  - VLM difficulty    vs human score disagreement  -> does the VLM's difficulty capture real spread?
  - VLM difficulty    vs human difficulty           -> does the VLM match the humans' difficulty?
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from calibration import dedup
from common import load_run, write_json


def main() -> None:
    df = dedup(load_run("eva_full").df)
    g = df.groupby("imageName").agg(
        vlm_diff=("pred_difficulty", "mean"),
        human_diff=("gt_difficulty", "mean"),
        human_score_std=("gt_score", "std"),
        n=("gt_score", "size")).dropna()
    g = g[g["n"] >= 10]

    def sp(a, b):
        return round(float(stats.spearmanr(g[a], g[b]).statistic), 4)

    report = {
        "dataset": "EVA", "n_images": int(len(g)),
        "human_difficulty_vs_disagreement": sp("human_diff", "human_score_std"),
        "vlm_difficulty_vs_disagreement": sp("vlm_diff", "human_score_std"),
        "vlm_difficulty_vs_human_difficulty": sp("vlm_diff", "human_diff"),
        "note": "higher difficulty should mean MORE disagreement; sign>0 = valid.",
    }
    write_json(report, "validity_difficulty.json")
    print("\nWrote results/validity_difficulty.json\n")
    print(f"EVA difficulty validity (Spearman over {len(g)} images):")
    print(f"  human difficulty  vs human disagreement : {report['human_difficulty_vs_disagreement']:+.3f}")
    print(f"  VLM   difficulty  vs human disagreement : {report['vlm_difficulty_vs_disagreement']:+.3f}")
    print(f"  VLM   difficulty  vs human difficulty   : {report['vlm_difficulty_vs_human_difficulty']:+.3f}")


if __name__ == "__main__":
    main()
