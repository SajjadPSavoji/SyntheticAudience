"""C4 deliverables — auto-refinement trajectory analysis (research_plan.md sec.8.4).

Reads the per-step logs written by ``script/c4_refine.py`` under
``data/results/c4_<condition>/`` and produces the C4 figures + report:

  1. Best-so-far aesthetic vs step (1-10), one line/condition, bootstrap-CI bands.
  2. Trajectory-AUC per condition (+ the society-vs-blind paired diff) — the
     pre-registered endpoint.
  3. Convergence step (first step reaching >=90% of the total best-so-far gain).
  4. Drift-of-best vs step (verifies the guardrail held; the oracle rides the cap).
  5. Complaint diversity per step (society vs blind) — the mechanism evidence.

Pure re-analysis: no GPU/inference. Run from ``scripts/analysis/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REPO, _json_default

# Same default root as script/c4_refine.py; override with --output-root.
DEFAULT_OUTPUT_ROOT = os.path.join(REPO, "outputs", "c4_auto_research")
CONDITIONS = ["static", "blind", "society", "reward_only"]
COLORS = {"static": "#9e9e9e", "blind": "#1f77b4", "society": "#d62728",
          "reward_only": "#2ca02c"}
LABELS = {"static": "static string", "blind": "blind VLM", "society": "society",
          "reward_only": "reward-only (oracle)"}
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_c4(condition: str, logs_dir: str) -> pd.DataFrame:
    """Concatenate all part-files across shards for one c4_<condition> run."""
    run = f"c4_{condition}"
    parts = sorted(glob.glob(os.path.join(logs_dir, run, f"{run}*.part-*.json")))
    rows: list[dict] = []
    for p in parts:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return pd.DataFrame(rows)


def _best_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """[n_images, n_steps] best-so-far objective, ffilled (monotone) over steps."""
    piv = df.pivot_table(index="image_id", columns="step", values="best_obj", aggfunc="first")
    piv = piv.reindex(columns=range(int(df["step"].max()) + 1))
    piv = piv.ffill(axis=1).dropna(axis=0)
    return piv.to_numpy(dtype=float), list(piv.index), list(piv.columns)


def _boot_ci(vals: np.ndarray, stat=np.mean, n=N_BOOT) -> list[float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    draws = stat(vals[idx], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _auc(M: np.ndarray) -> np.ndarray:
    """Per-image normalized area under the best-so-far curve (avg height).

    Manual dx=1 trapezoid (version-proof: np.trapz was removed in NumPy 2.0).
    """
    R = M.shape[1] - 1
    area = ((M[:, 1:] + M[:, :-1]) / 2.0).sum(axis=1)
    return area / max(R, 1)


def _convergence(M: np.ndarray) -> np.ndarray:
    """Per-image first step reaching >=90% of its total gain (NaN if no gain)."""
    total = M[:, -1] - M[:, 0]
    out = np.full(M.shape[0], np.nan)
    for i in range(M.shape[0]):
        if total[i] <= 1e-9:
            continue
        thr = M[i, 0] + 0.9 * total[i]
        hit = np.where(M[i] >= thr)[0]
        if len(hit):
            out[i] = hit[0]
    return out


def _norm_complaint(c: str) -> str:
    return " ".join(str(c).lower().strip().rstrip(".").split())


def _finals(df: pd.DataFrame) -> dict:
    """Per-image final outcome: gain over the source + the committed best's drift."""
    out: dict = {}
    for img, recs in df.groupby("image_id"):
        recs = recs.sort_values("step")
        start, final = float(recs.iloc[0]["best_obj"]), float(recs.iloc[-1]["best_obj"])
        out[str(img)] = {"start": start, "final": final, "gain": final - start,
                         "drift_final": float(recs.iloc[-1]["drift_of_best"])}
    return out


def analyze(logs_dir: str, analysis_dir: str) -> dict:
    figs = os.path.join(analysis_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    data = {c: load_c4(c, logs_dir) for c in CONDITIONS}
    present = [c for c in CONDITIONS if len(data[c])]
    if not present:
        raise SystemExit(f"No c4 logs found under {logs_dir}/c4_* — run script/c4_refine.py first.")

    report: dict = {"conditions": present, "n_boot": N_BOOT}
    mats: dict = {}

    # 1) best-so-far trajectory + CI bands -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    traj: dict = {}
    for c in present:
        M, _, steps = _best_matrix(data[c])
        mats[c] = M
        mean = M.mean(0)
        ci = np.array([_boot_ci(M[:, s]) for s in range(M.shape[1])])
        traj[c] = {"steps": steps, "mean": mean.tolist(),
                   "ci_lo": ci[:, 0].tolist(), "ci_hi": ci[:, 1].tolist(),
                   "n_images": int(M.shape[0])}
        ax.plot(steps, mean, "-o", ms=3, color=COLORS[c], label=LABELS[c])
        ax.fill_between(steps, ci[:, 0], ci[:, 1], color=COLORS[c], alpha=0.15)
    ax.set_xlabel("refinement step")
    ax.set_ylabel("best-so-far aesthetic score")
    ax.set_title("C4 — audience-guided editing: best-so-far vs step")
    ax.legend(frameon=False)
    fig.tight_layout()
    traj_path = os.path.join(figs, "c4_trajectory.png")
    fig.savefig(traj_path, dpi=130)
    plt.close(fig)
    report["trajectory"] = traj
    report["_figure_trajectory"] = os.path.relpath(traj_path, analysis_dir)

    # 2) trajectory-AUC (+ society vs blind paired diff) ---------------------
    auc: dict = {}
    per_img_auc: dict = {}
    for c in present:
        a = _auc(mats[c])
        per_img_auc[c] = pd.Series(a, index=_best_matrix(data[c])[1])
        auc[c] = {"mean": float(a.mean()), "ci": _boot_ci(a), "n_images": int(len(a))}
    report["trajectory_auc"] = auc

    if "society" in present and "blind" in present:
        common_ids = per_img_auc["society"].index.intersection(per_img_auc["blind"].index)
        diff = (per_img_auc["society"].loc[common_ids]
                - per_img_auc["blind"].loc[common_ids]).to_numpy()
        report["society_vs_blind_auc"] = {
            "mean_diff": float(diff.mean()), "ci": _boot_ci(diff),
            "n_images": int(len(diff)),
            "wins": int((diff > 0).sum()), "total": int(len(diff)),
        }

    # 3) convergence step ----------------------------------------------------
    conv: dict = {}
    for c in present:
        cs = _convergence(mats[c])
        valid = cs[~np.isnan(cs)]
        conv[c] = {"median_step": (float(np.median(valid)) if len(valid) else None),
                   "frac_improved": float(np.mean(~np.isnan(cs)))}
    report["convergence"] = conv

    # 4) drift-of-best vs step ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.0))
    drift: dict = {}
    for c in present:
        g = data[c].groupby("step")["drift_of_best"].mean()
        drift[c] = {"steps": g.index.tolist(), "mean_drift": g.values.tolist()}
        ax.plot(g.index, g.values, "-o", ms=3, color=COLORS[c], label=LABELS[c])
    ax.axhline(0.85, ls="--", c="k", lw=0.8, label="drift cap (0.85)")
    ax.set_xlabel("refinement step")
    ax.set_ylabel("identity similarity of best (DINOv2)")
    ax.set_title("C4 — drift guardrail: identity retention vs step")
    ax.legend(frameon=False)
    fig.tight_layout()
    drift_path = os.path.join(figs, "c4_drift.png")
    fig.savefig(drift_path, dpi=130)
    plt.close(fig)
    report["drift"] = drift
    report["_figure_drift"] = os.path.relpath(drift_path, analysis_dir)

    # 5) complaint diversity per step (society vs blind) ---------------------
    div: dict = {}
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for c in ("society", "blind"):
        if c not in present:
            continue
        d = data[c][data[c]["step"] > 0].copy()
        d["n_unique"] = d["complaints"].apply(
            lambda cs: len({_norm_complaint(x) for x in (cs or []) if str(x).strip()}))
        g = d.groupby("step")["n_unique"].mean()
        div[c] = {"steps": g.index.tolist(), "mean_unique_complaints": g.values.tolist(),
                  "overall_mean": float(d["n_unique"].mean())}
        ax.plot(g.index, g.values, "-o", ms=3, color=COLORS[c], label=LABELS[c])
    ax.set_xlabel("refinement step")
    ax.set_ylabel("distinct complaints per step")
    ax.set_title("C4 — feedback diversity: society vs blind VLM")
    ax.legend(frameon=False)
    fig.tight_layout()
    div_path = os.path.join(figs, "c4_diversity.png")
    fig.savefig(div_path, dpi=130)
    plt.close(fig)
    report["complaint_diversity"] = div
    report["_figure_diversity"] = os.path.relpath(div_path, analysis_dir)

    # 6) headline: final gain, pairwise win-rate, and the drift-vs-gain check
    finals = {c: _finals(data[c]) for c in present}
    headline: dict = {}
    for c in present:
        g = np.array([finals[c][i]["gain"] for i in finals[c]])
        d = np.array([finals[c][i]["drift_final"] for i in finals[c]])
        headline[c] = {"n_images": int(len(g)),
                       "mean_gain": float(g.mean()), "gain_ci": _boot_ci(g),
                       "mean_drift_final": float(np.nanmean(d))}
    report["headline"] = headline

    # pairwise win-rate: fraction of images where A's final best beats B's.
    winrates: dict = {}
    for a in present:
        for b in present:
            if a == b:
                continue
            ids = set(finals[a]) & set(finals[b])
            if not ids:
                continue
            wins = sum(finals[a][i]["final"] > finals[b][i]["final"] for i in ids)
            winrates[f"{a}_vs_{b}"] = {"win_rate": wins / len(ids),
                                      "wins": int(wins), "n": len(ids)}
    report["win_rates"] = winrates

    # headline figure: (left) mean final gain per condition; (right) drift-vs-gain
    # scatter — the honesty check that gains aren't just identity drift.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = list(range(len(present)))
    means = [headline[c]["mean_gain"] for c in present]
    errs = [[headline[c]["mean_gain"] - headline[c]["gain_ci"][0] for c in present],
            [headline[c]["gain_ci"][1] - headline[c]["mean_gain"] for c in present]]
    axL.bar(xs, means, color=[COLORS[c] for c in present])
    axL.errorbar(xs, means, yerr=errs, fmt="none", ecolor="k", capsize=4, lw=1)
    axL.set_xticks(xs)
    axL.set_xticklabels([LABELS[c] for c in present], rotation=15, ha="right")
    axL.set_ylabel("mean final aesthetic gain")
    axL.set_title("C4 — improvement over source (95% CI)")
    axL.axhline(0, c="k", lw=0.6)
    for c in present:
        gx = [finals[c][i]["drift_final"] for i in finals[c]]
        gy = [finals[c][i]["gain"] for i in finals[c]]
        axR.scatter(gx, gy, color=COLORS[c], label=LABELS[c], s=40, alpha=0.8, edgecolor="w")
    axR.axvline(0.85, ls="--", c="k", lw=0.8, label="drift cap")
    axR.set_xlabel("identity similarity of final best (DINOv2)")
    axR.set_ylabel("final aesthetic gain")
    axR.set_title("C4 — gain vs identity drift (real improvement vs transformation)")
    axR.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    head_path = os.path.join(figs, "c4_headline.png")
    fig.savefig(head_path, dpi=130)
    plt.close(fig)
    report["_figure_headline"] = os.path.relpath(head_path, analysis_dir)

    # markdown summary table (the paper's main results table)
    _write_summary_table(os.path.join(analysis_dir, "c4_summary.md"),
                         present, headline, auc, conv, winrates, div)
    report["_table_summary"] = "c4_summary.md"

    return report


def _write_summary_table(path, present, headline, auc, conv, winrates, div) -> None:
    def wr(a, b):
        r = winrates.get(f"{a}_vs_{b}")
        return f"{r['win_rate']:.0%} ({r['wins']}/{r['n']})" if r else "—"
    lines = [
        "# C4 — auto-refinement results summary",
        "",
        "| condition | n | mean final gain | trajectory-AUC | conv. step | mean drift(final) | uniq complaints/step | win vs static | win vs blind |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in present:
        h, a = headline[c], auc[c]
        gci = h["gain_ci"]
        cs = conv[c]["median_step"]
        dv = div.get(c, {}).get("overall_mean")
        dv_str = f"{dv:.2f}" if dv is not None else "—"
        lines.append(
            f"| {LABELS[c]} | {h['n_images']} | "
            f"{h['mean_gain']:+.3f} [{gci[0]:+.3f}, {gci[1]:+.3f}] | "
            f"{a['mean']:.3f} | {cs if cs is not None else '—'} | "
            f"{h['mean_drift_final']:.3f} | {dv_str} | "
            f"{wr(c, 'static')} | {wr(c, 'blind')} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 trajectory deliverables.")
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                    help="same root passed to script/c4_refine.py (default: %(default)s).")
    ap.add_argument("--logs-dir", default=None, help="override <root>/logs.")
    ap.add_argument("--analysis-dir", default=None, help="override <root>/analysis.")
    args = ap.parse_args()
    logs_dir = args.logs_dir or os.path.join(args.output_root, "logs")
    analysis_dir = args.analysis_dir or os.path.join(args.output_root, "analysis")

    report = analyze(logs_dir, analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)
    out_json = os.path.join(analysis_dir, "c4.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"wrote {out_json}")
    if "society_vs_blind_auc" in report:
        s = report["society_vs_blind_auc"]
        print(f"society vs blind trajectory-AUC diff: {s['mean_diff']:+.4f} "
              f"CI{s['ci']}  ({s['wins']}/{s['total']} images)")


if __name__ == "__main__":
    main()
