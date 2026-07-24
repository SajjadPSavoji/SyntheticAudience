"""C3 — cross-cultural preference on AI-generated images (Rapidata).

Analyzes the `rapidata_persona` run: per-vote pairwise-choice predictions
(VLM role-playing the voter's country) against real human votes on
dalle-3 vs flux image pairs, each vote tagged with the voter's country.

Tests whether the LAPIS nationality signal (C1 between-group separation +0.17
on real art) *transfers* to generated images. Produces:
  - overall predictive accuracy vs a majority-class baseline (+ order-bias audit),
  - country-level human-vs-VLM flux-preference scatter,
  - the pre-registered PAIR-CONTROLLED between-country separation (the metric
    comparable to LAPIS C1) with a bootstrap CI,
  - a real->generated transfer bar (LAPIS C1 vs Rapidata C3).

Run from scripts/analysis/. Reads data/results/rapidata_persona/, writes
results/c3.json + results/figs/c3_*.png. No GPU.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, REPO, ensure_out, write_json

RUN_DIR = os.path.join(REPO, "data", "results", "rapidata_persona")
LAPIS_C1_SEPARATION = 0.17   # research_plan.md sec.14.11c (real-art nationality)
MIN_COUNTRY_VOTES = 50       # for the country-level scatter
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_votes() -> list[dict]:
    recs: list[dict] = []
    for p in sorted(glob.glob(os.path.join(RUN_DIR, "*.part-*.json"))):
        with open(p, encoding="utf-8") as f:
            recs += json.load(f)
    # image1 = model1 (dalle-3), image2 = model2 (flux); choices are canonical 1/2.
    return [r for r in recs if r.get("pred_choice") in (1, 2) and r.get("human_choice") in (1, 2)]


def analyze() -> dict:
    V = load_votes()
    h = np.array([r["human_choice"] for r in V])
    p = np.array([r["pred_choice"] for r in V])
    sf = np.array([r["shown_first"] for r in V])
    rep: dict = {"n_votes": len(V),
                 "n_pairs": len({r["pair_id"] for r in V}),
                 "n_countries": len({r["country"] for r in V}),
                 "models": sorted({r["model1"] for r in V} | {r["model2"] for r in V})}

    # 1) overall accuracy vs baselines + order bias -------------------------
    acc = float(np.mean(h == p))
    human_flux = float(np.mean(h == 2))           # image2 = flux
    vlm_flux = float(np.mean(p == 2))
    majority = max(human_flux, 1 - human_flux)     # always-pick-the-more-liked-model
    order_bias = float(np.mean(p == sf))           # 0.5 = no position bias
    rep["accuracy"] = {"vlm": acc, "majority_baseline": majority, "chance": 0.5,
                       "human_flux_rate": human_flux, "vlm_flux_rate": vlm_flux,
                       "order_bias_pick_first_shown": order_bias}

    # 2) country-level flux-preference (human vs VLM) -----------------------
    by_c = defaultdict(list)
    for r in V:
        by_c[r["country"]].append((r["human_choice"] == 2, r["pred_choice"] == 2))
    ctab = []
    for c, lst in by_c.items():
        if len(lst) < MIN_COUNTRY_VOTES:
            continue
        hh = np.array([x[0] for x in lst], float)
        pp = np.array([x[1] for x in lst], float)
        ctab.append({"country": c, "n": len(lst),
                     "human_flux": float(hh.mean()), "vlm_flux": float(pp.mean()),
                     "acc": float(np.mean([a == b for a, b in
                                           zip([r["human_choice"] for r in V if r["country"] == c],
                                               [r["pred_choice"] for r in V if r["country"] == c])]))})
    ctab.sort(key=lambda d: -d["n"])
    hf = np.array([d["human_flux"] for d in ctab])
    vf = np.array([d["vlm_flux"] for d in ctab])
    w = np.array([d["n"] for d in ctab], float)
    naive = float(np.corrcoef(hf, vf)[0, 1]) if len(ctab) > 2 else float("nan")
    rep["country_table"] = ctab
    rep["naive_cross_country_corr"] = naive  # confounded by different pairs/country

    # 3) PAIR-CONTROLLED between-country separation (the C1-comparable metric)
    cell_h, cell_p = defaultdict(list), defaultdict(list)
    for r in V:
        cell_h[(r["pair_id"], r["country"])].append(r["human_choice"] == 2)
        cell_p[(r["pair_id"], r["country"])].append(r["pred_choice"] == 2)
    by_pair = defaultdict(list)
    for k in cell_h:
        by_pair[k[0]].append((float(np.mean(cell_h[k])), float(np.mean(cell_p[k]))))
    pair_items = [(pid, v) for pid, v in by_pair.items() if len(v) >= 2]

    def sep_from(items):
        a, b = [], []
        for _, v in items:
            hm = np.mean([x[0] for x in v]); pm = np.mean([x[1] for x in v])
            for hv, pv in v:
                a.append(hv - hm); b.append(pv - pm)
        a, b = np.array(a), np.array(b)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    sep = sep_from(pair_items)
    boot = []
    for _ in range(N_BOOT):
        samp = [pair_items[i] for i in RNG.integers(0, len(pair_items), len(pair_items))]
        s = sep_from(samp)
        if not np.isnan(s):
            boot.append(s)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    n_cells = sum(len(v) for _, v in pair_items)
    rep["pair_controlled_separation"] = {"value": sep, "ci": ci,
                                         "n_pairs": len(pair_items), "n_cells": n_cells}

    # 4) AGGREGATION TRANSFER (the reframed C3 headline) --------------------
    # Per pair, pool ALL votes: does the panel aggregate track the crowd, even
    # though individual votes are near-chance? (C2 mechanism on generated images.)
    pool_h, pool_p = defaultdict(list), defaultdict(list)
    for r in V:
        pool_h[r["pair_id"]].append(r["human_choice"] == 2)   # image2 = flux
        pool_p[r["pair_id"]].append(r["pred_choice"] == 2)
    keys = list(pool_h)
    Hn = np.array([np.mean(pool_h[k]) for k in keys])   # human flux-winrate / pair
    Pn = np.array([np.mean(pool_p[k]) for k in keys])   # VLM panel flux-winrate / pair
    Nn = np.array([len(pool_h[k]) for k in keys])

    def _bootci(fn, n=N_BOOT):
        vals = [fn(RNG.integers(0, len(keys), len(keys))) for _ in range(n)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    corr = float(np.corrcoef(Hn, Pn)[0, 1])
    corr_ci = _bootci(lambda idx: np.corrcoef(Hn[idx], Pn[idx])[0, 1])
    hmaj = Hn > 0.5
    agg_acc = float(np.mean(hmaj == (Pn > 0.5)))
    maj = float(max(hmaj.mean(), 1 - hmaj.mean()))
    lift_ci = _bootci(lambda idx: np.mean(hmaj[idx] == (Pn[idx] > 0.5))
                      - max(hmaj[idx].mean(), 1 - hmaj[idx].mean()))
    ind_acc = float(np.mean([r["human_choice"] == r["pred_choice"] for r in V]))

    # N-curve: subsample N VLM votes/pair -> panel-majority vs human pair-majority
    ncurve = {}
    for Nsub in (1, 2, 5, 10, 20):
        accs = []
        for _ in range(200):
            ok = tot = 0
            for k in keys:
                d = pool_p[k]
                if len(d) < Nsub:
                    continue
                sub = RNG.choice(d, size=Nsub, replace=False)
                m = float(np.mean(sub))
                pv = (m > 0.5) if m != 0.5 else (RNG.random() < 0.5)
                ok += int(pv == (np.mean(pool_h[k]) > 0.5)); tot += 1
            accs.append(ok / tot)
        ncurve[Nsub] = float(np.mean(accs))
    rep["aggregation"] = {
        "pair_corr": corr, "pair_corr_ci": corr_ci,
        "individual_acc": ind_acc, "aggregate_acc": agg_acc, "aggregate_acc_majority": maj,
        "aggregate_minus_majority_ci": lift_ci,
        "n_curve": ncurve, "votes_per_pair_median": float(np.median(Nn))}

    _figures(rep, ctab, hf, vf, w, sep, ci)
    _agg_figures(rep, Hn, Pn, Nn, ncurve)
    return rep


def _agg_figures(rep, Hn, Pn, Nn, ncurve):
    figs = os.path.join(OUT_DIR, "figs")
    os.makedirs(figs, exist_ok=True)
    ag = rep["aggregation"]

    # Fig 4: aggregate scatter — VLM panel winrate vs human crowd winrate per pair
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(Hn, Pn, s=6, alpha=0.35, color="#1f77b4")
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.7)
    ax.set_xlabel("human crowd flux-winrate (per pair)")
    ax.set_ylabel("VLM panel flux-winrate (per pair)")
    ax.set_title(f"C3 (reframed) — aggregate tracks the crowd\n"
                 f"r = {ag['pair_corr']:+.2f} [{ag['pair_corr_ci'][0]:+.2f}, {ag['pair_corr_ci'][1]:+.2f}]"
                 f"  over {len(Hn)} generated pairs")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_aggregate_scatter.png"), dpi=130); plt.close(fig)

    # Fig 5: the N-curve — aggregation mechanism transfers to generated images
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    xs = sorted(ncurve); ys = [ncurve[n] for n in xs]
    ax.plot(xs, ys, "-o", color="#d62728", label="panel aggregate")
    ax.axhline(ag["aggregate_acc_majority"], ls="--", c="#7f7f7f", label="always-flux majority")
    ax.axhline(ag["individual_acc"], ls=":", c="k", label="single vote (individual)")
    ax.set_xlabel("panel size N (personas aggregated)")
    ax.set_ylabel("accuracy predicting the crowd's preferred image")
    ax.set_title("C3 (reframed) — the aggregation mechanism transfers\nto AI-generated images")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_ncurve.png"), dpi=130); plt.close(fig)
    rep.setdefault("_figures", []).extend(["figs/c3_aggregate_scatter.png", "figs/c3_ncurve.png"])


def _figures(rep, ctab, hf, vf, w, sep, ci):
    figs = os.path.join(OUT_DIR, "figs")
    os.makedirs(figs, exist_ok=True)
    A = rep["accuracy"]

    # Fig 1: accuracy vs baselines
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ["VLM (persona)", "always-flux", "chance"]
    vals = [A["vlm"], A["majority_baseline"], A["chance"]]
    ax.bar(bars, vals, color=["#d62728", "#7f7f7f", "#cccccc"])
    ax.axhline(0.5, c="k", lw=0.6)
    ax.set_ylim(0.45, max(0.6, max(vals) + 0.03))
    ax.set_ylabel("pairwise accuracy")
    ax.set_title(f"C3 — generated-image preference\n(order bias: picks first-shown {A['order_bias_pick_first_shown']:.2f})")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_accuracy.png"), dpi=130); plt.close(fig)

    # Fig 2: country scatter human vs VLM flux-preference
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(hf, vf, s=np.sqrt(w) * 6, alpha=0.6, color="#1f77b4", edgecolor="w")
    for d in ctab[:12]:
        ax.annotate(d["country"], (d["human_flux"], d["vlm_flux"]), fontsize=7,
                    xytext=(2, 2), textcoords="offset points")
    lim = [min(hf.min(), vf.min()) - 0.02, max(hf.max(), vf.max()) + 0.02]
    ax.plot(lim, lim, ls="--", c="k", lw=0.7)
    ax.set_xlabel("human flux-preference rate (per country)")
    ax.set_ylabel("VLM predicted flux-preference rate")
    ax.set_title(f"C3 — does country-conditioning track real taste?\n"
                 f"naive corr = {rep['naive_cross_country_corr']:+.2f} (confounded by pair-mix)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_country_scatter.png"), dpi=130); plt.close(fig)

    # Fig 3: real->generated transfer (LAPIS C1 vs Rapidata C3)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = ["LAPIS C1\n(real art)", "Rapidata C3\n(generated)"]
    ys = [LAPIS_C1_SEPARATION, sep]
    err = [[0, sep - ci[0]], [0, ci[1] - sep]]
    ax.bar(xs, ys, color=["#2ca02c", "#d62728"])
    ax.errorbar([1], [sep], yerr=[[sep - ci[0]], [ci[1] - sep]], fmt="none", ecolor="k", capsize=5)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_ylabel("between-group separation\n(corr of predicted vs observed group gaps)")
    ax.set_title("C3 — cross-cultural signal does NOT transfer\nfrom real art to generated images")
    for i, v in enumerate(ys):
        ax.text(i, v + 0.005, f"{v:+.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_transfer.png"), dpi=130); plt.close(fig)

    rep["_figures"] = ["figs/c3_accuracy.png", "figs/c3_country_scatter.png", "figs/c3_transfer.png"]


def main() -> None:
    ensure_out()
    rep = analyze()
    path = write_json(rep, "c3.json")
    A = rep["accuracy"]; S = rep["pair_controlled_separation"]; G = rep["aggregation"]
    print(f"wrote {path}")
    print(f"votes={rep['n_votes']}  pairs={rep['n_pairs']}  countries={rep['n_countries']}  "
          f"models={rep['models']}")
    print("\n== C3a (reframed HEADLINE): aggregation transfers to generated images ==")
    print(f"  aggregate corr(panel vs crowd): r={G['pair_corr']:+.3f} "
          f"CI[{G['pair_corr_ci'][0]:+.3f},{G['pair_corr_ci'][1]:+.3f}]")
    print(f"  individual acc {G['individual_acc']:.3f} -> aggregate acc {G['aggregate_acc']:.3f} "
          f"(majority {G['aggregate_acc_majority']:.3f}; lift CI "
          f"[{G['aggregate_minus_majority_ci'][0]:+.3f},{G['aggregate_minus_majority_ci'][1]:+.3f}])")
    print(f"  N-curve: " + "  ".join(f"N={n}:{G['n_curve'][n]:.3f}" for n in sorted(G['n_curve'])))
    print("\n== C3b (honest negative): cross-cultural differentiation does NOT transfer ==")
    print(f"  pair-controlled between-country separation: {S['value']:+.3f} "
          f"CI[{S['ci'][0]:+.3f},{S['ci'][1]:+.3f}]  (LAPIS C1 was +{LAPIS_C1_SEPARATION})")
    print(f"  base pairwise acc {A['vlm']:.3f} vs always-flux {A['majority_baseline']:.3f}; "
          f"order-bias {A['order_bias_pick_first_shown']:.2f}")


if __name__ == "__main__":
    main()
