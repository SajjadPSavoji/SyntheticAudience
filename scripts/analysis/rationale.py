"""#7 Rationale analysis — persona signal in the free text (bypasses score quantization).

At temperature 0 the integer score often collapses to one value across personas, but the
free-text rationale still varies (we saw 25 personas -> 1 score but 13 distinct comments). So the
persona IS steering the model; the argmax just hides it in the number. We test that directly:

  Probe: can a rater attribute be predicted from THEIR rationale text alone? Train TF-IDF +
  logistic regression to predict a binarized persona attribute from the comment, report ROC-AUC
  (3-fold). For the persona run this should beat 0.5; for the blind run (generic prompt, no
  persona) it should sit at chance. A full-vs-blind AUC gap = persona leaking into the language.
  The probe is also re-scored with folds split by rater (GroupKFold on userId, TF-IDF fit inside
  each fold), so no rater's comments appear in both train and test: a check that the probe learns
  the attribute rather than recognizing individual raters' persona phrasing.

Also: within-image rationale diversity (distinct comments / raters) for full vs blind, and the
words most predictive of a LOW rating (a peek at the "complaints" C4 will mine).

Model-free beyond TF-IDF + logistic regression (no downloads). Subsampled for speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

from attrs import load_attributes
from calibration import dedup
from common import PRIMARY_DIM, load_run, write_json

# (attribute, how to binarize) per dataset — a coarse, robust probe target
TARGET = {"PARA": "artExperience", "EVA": "photographic_level", "LAPIS": "art_interest"}
SUBSAMPLE = 30000
RNG = np.random.default_rng(0)


MIN_CLASS = 500


def _target(df: pd.DataFrame, attr: str) -> pd.Series:
    """Integer class labels for the probe: terciles if numeric, factorized levels if
    categorical. Rare classes (< MIN_CLASS) are dropped (returned as NaN)."""
    num = pd.to_numeric(df[attr], errors="coerce")
    if num.notna().mean() > 0.9 and num.nunique() > 6:
        y = pd.qcut(num, 3, labels=False, duplicates="drop")
        y = pd.Series(y, index=df.index).astype("float")
    else:
        codes, _ = pd.factorize(df[attr])
        y = pd.Series(codes, index=df.index).astype("float").where(lambda x: x >= 0)
    counts = y.value_counts()
    keep = counts[counts >= MIN_CLASS].index
    return y.where(y.isin(keep))


def _auc(text: pd.Series, y: pd.Series) -> float:
    m = y.notna()
    text, y = text[m], y[m].astype(int)
    if y.nunique() < 2:
        return float("nan")
    vec = TfidfVectorizer(max_features=2000, stop_words="english", min_df=5)
    X = vec.fit_transform(text.fillna(""))
    clf = LogisticRegression(max_iter=300, C=1.0)
    scoring = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr"
    return float(np.mean(cross_val_score(clf, X, y, cv=3, scoring=scoring)))


def _auc_by_rater(text: pd.Series, y: pd.Series, raters: pd.Series) -> float:
    """Same probe as `_auc`, but folds are split by rater and TF-IDF is fit inside each fold."""
    m = y.notna()
    text, y, raters = text[m].fillna(""), y[m].astype(int), raters[m]
    if y.nunique() < 2:
        return float("nan")
    pipe = make_pipeline(TfidfVectorizer(max_features=2000, stop_words="english", min_df=5),
                         LogisticRegression(max_iter=300, C=1.0))
    scoring = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr"
    return float(np.mean(cross_val_score(pipe, text, y, groups=raters, cv=GroupKFold(n_splits=3),
                                         scoring=scoring)))


def _auc_by_rater_ci(text: pd.Series, y: pd.Series, raters: pd.Series,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Rater-split AUC (mean over folds, as in `_auc_by_rater`) with a 95% bootstrap CI.

    Each fold's model scores only its own held-out raters, and the AUC is computed within
    each fold and then averaged; pooling probabilities across folds would mix differently
    calibrated models and bias the AUC down. For the CI, raters are resampled with
    replacement (each rater stays in its fold) and the per-fold AUCs are recomputed on
    the fixed held-out predictions, so the interval reflects which raters were sampled."""
    m = y.notna()
    text, y, raters = text[m].fillna(""), y[m].astype(int).to_numpy(), raters[m].to_numpy()
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    fold = np.zeros(len(y), dtype=int)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=3).split(text, y, groups=raters)):
        pipe = make_pipeline(TfidfVectorizer(max_features=2000, stop_words="english", min_df=5),
                             LogisticRegression(max_iter=300, C=1.0))
        pipe.fit(text.iloc[tr], y[tr])
        proba[te] = pipe.predict_proba(text.iloc[te])
        fold[te] = f

    def auc(rows):
        yy, pp = y[rows], proba[rows]
        if len(np.unique(yy)) < len(classes):
            return np.nan
        if len(classes) == 2:
            return roc_auc_score(yy, pp[:, 1])
        return roc_auc_score(yy, pp, multi_class="ovr", labels=classes)

    def mean_fold_auc(rows):
        return np.nanmean([auc(rows[fold[rows] == f]) for f in range(3)])

    by_rater = {}
    for i, r in enumerate(raters):
        by_rater.setdefault(r, []).append(i)
    keys = np.array(list(by_rater), dtype=object)
    rng = np.random.default_rng(seed)
    boot = np.array([mean_fold_auc(np.concatenate([by_rater[k] for k in keys[rng.integers(0, len(keys), len(keys))]]))
                     for _ in range(n_boot)])
    return {"n_comments": int(len(y)), "n_raters": int(len(keys)),
            "auc": round(float(mean_fold_auc(np.arange(len(y)))), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                     round(float(np.percentile(boot, 97.5)), 4)]}


def _diversity_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> list:
    """95% bootstrap CI over images for `_diversity` (same image filter)."""
    g = df.groupby("imageName")["comment"].agg(["nunique", "size"])
    x = (g["nunique"] / g["size"])[g["size"] >= 10].to_numpy()
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _diversity_n(df: pd.DataFrame) -> int:
    """Number of images entering `_diversity` (those with at least 10 raters)."""
    return int((df.groupby("imageName")["comment"].size() >= 10).sum())


def _diversity(df: pd.DataFrame) -> float:
    g = df.groupby("imageName")["comment"].agg(["nunique", "size"])
    g = g[g["size"] >= 10]
    return float((g["nunique"] / g["size"]).mean())


def _low_words(df: pd.DataFrame, dim: str) -> list:
    y = (df[f"{dim}_gt_norm"] < df[f"{dim}_gt_norm"].median()).astype(int)
    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=10)
    X = vec.fit_transform(df["comment"].fillna(""))
    clf = LogisticRegression(max_iter=300).fit(X, y)  # y=1 => low rating
    terms = np.array(vec.get_feature_names_out())
    top = terms[np.argsort(clf.coef_[0])[-12:]][::-1]
    return list(top)


def analyze(dataset: str) -> dict:
    dim = PRIMARY_DIM[dataset]
    full = dedup(load_run(f"{dataset.lower()}_full").df).copy()
    blind = dedup(load_run(f"{dataset.lower()}_blind").df).copy()
    for d in (full, blind):
        d["userId"] = d["userId"].astype(str)
    attrs = load_attributes(dataset)
    full = full.join(attrs, on="userId")
    blind = blind.join(attrs, on="userId")

    attr = TARGET[dataset]
    sub = full.dropna(subset=["comment", attr])
    if len(sub) > SUBSAMPLE:
        sub = sub.sample(SUBSAMPLE, random_state=0)
    auc_full = _auc(sub["comment"], _target(sub, attr))
    auc_full_rater = _auc_by_rater(sub["comment"], _target(sub, attr), sub["userId"])
    oof_full = _auc_by_rater_ci(sub["comment"], _target(sub, attr), sub["userId"])

    subb = blind.dropna(subset=["comment", attr])
    if len(subb) > SUBSAMPLE:
        subb = subb.sample(SUBSAMPLE, random_state=0)
    auc_blind = _auc(subb["comment"], _target(subb, attr))
    auc_blind_rater = _auc_by_rater(subb["comment"], _target(subb, attr), subb["userId"])
    oof_blind = _auc_by_rater_ci(subb["comment"], _target(subb, attr), subb["userId"])

    return {"dataset": dataset, "probe_attribute": attr,
            "auc_full": round(auc_full, 4), "auc_blind": round(auc_blind, 4),
            "auc_gain_full_over_blind": round(auc_full - auc_blind, 4),
            "auc_full_rater_split": round(auc_full_rater, 4),
            "auc_blind_rater_split": round(auc_blind_rater, 4),
            "auc_full_rater_oof": oof_full["auc"], "auc_full_rater_oof_ci95": oof_full["ci95"],
            "auc_blind_rater_oof": oof_blind["auc"], "auc_blind_rater_oof_ci95": oof_blind["ci95"],
            "probe_n_comments_full": oof_full["n_comments"], "probe_n_raters_full": oof_full["n_raters"],
            "probe_n_comments_blind": oof_blind["n_comments"], "probe_n_raters_blind": oof_blind["n_raters"],
            "diversity_n_images_full": _diversity_n(full), "diversity_n_images_blind": _diversity_n(blind),
            "rationale_diversity_full_ci95": _diversity_ci(full),
            "rationale_diversity_blind_ci95": _diversity_ci(blind),
            "rationale_diversity_full": round(_diversity(full), 4),
            "rationale_diversity_blind": round(_diversity(blind), 4),
            "low_rating_words": _low_words(full, dim)}


def main() -> None:
    report = {ds: analyze(ds) for ds in TARGET}
    write_json(report, "rationale.json")
    print("\nWrote results/rationale.json\n")
    hdr = (f"{'dataset':7}{'probe attr':20}{'AUC full':>10}{'AUC blind':>11}"
           f"{'gain':>7}{'divF':>7}{'divB':>7}")
    print(hdr); print("-" * len(hdr))
    for ds, m in report.items():
        print(f"{ds:7}{m['probe_attribute']:20}{m['auc_full']:>10.3f}{m['auc_blind']:>11.3f}"
              f"{m['auc_gain_full_over_blind']:>7.3f}"
              f"{m['rationale_diversity_full']:>7.3f}{m['rationale_diversity_blind']:>7.3f}")
    print("\nAUC full > 0.5 and > AUC blind => the persona leaks into the rationale text (signal")
    print("the quantized score hides). divF/divB = distinct comments per rater, full vs blind.")
    for ds, m in report.items():
        print(f"{ds} rater-split AUC: full {m['auc_full_rater_split']:.3f}, "
              f"blind {m['auc_blind_rater_split']:.3f}")
    for ds, m in report.items():
        print(f"\n{ds} words most predictive of a LOW rating: {', '.join(m['low_rating_words'])}")


if __name__ == "__main__":
    main()
