"""userId -> structured rater attributes, joined from the source annotation files.

Result logs store the persona only as free text; to slice or measure steerability we need
the structured attributes back. This module returns one row per rater (indexed by string
userId) with named attribute columns, reading:
  PARA : para/annotation/PARA-UserInfo.csv
  EVA  : eva/data/users.csv           (birth year -> age; ids for region/gender/level)
  LAPIS: lapis/annotation/LAPIS_PIAA.csv  (demographics are inline, constant per participant)

Attribute columns are a mix of categorical (str) and numeric. `ATTRS[dataset]` lists them.
"""
from __future__ import annotations

import os

import pandas as pd

from common import REPO

DATA = os.path.join(REPO, "data")

ATTRS = {
    "PARA": ["age", "gender", "education", "artExperience", "photographyExperience",
             "big5_O", "big5_C", "big5_E", "big5_A", "big5_N"],
    "EVA": ["age", "region", "photographic_level", "gender", "eyecheck"],
    "LAPIS": ["age", "nationality", "gender", "education", "colorblind", "art_interest"],
}


def _para() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, "para", "annotation", "PARA-UserInfo.csv"))
    out = pd.DataFrame({
        "userId": d["userId"].astype(str),
        "age": d["age"], "gender": d["gender"], "education": d["EducationalLevel"],
        "artExperience": d["artExperience"],
        "photographyExperience": d["photographyExperience"],
        "big5_O": d["personality-O"], "big5_C": d["personality-C"],
        "big5_E": d["personality-E"], "big5_A": d["personality-A"],
        "big5_N": d["personality-N"],
    })
    return out.set_index("userId")


def _idstr(col: pd.Series) -> pd.Series:
    """Represent an id/code column as a clean string ('156.0'->'156'), tolerating
    multi-valued cells like '1,2' (kept verbatim)."""
    s = col.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def _eva() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, "eva", "data", "users.csv"), sep="=")
    out = pd.DataFrame({
        "userId": _idstr(d["id"]),
        "age": 2020 - pd.to_numeric(d["age"], errors="coerce"),  # birth year -> approx age
        "region": _idstr(d["region"]),
        "photographic_level": _idstr(d["photographic_level_id"]),
        "gender": _idstr(d["gender_id"]),
        "eyecheck": _idstr(d["eyecheck"]),
    })
    return out.drop_duplicates("userId").set_index("userId")


def _lapis() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, "lapis", "annotation", "LAPIS_PIAA.csv"))
    out = pd.DataFrame({
        "userId": d["participant_id"].astype(str),
        "age": pd.to_numeric(d["age"], errors="coerce"),
        "nationality": d["nationality"].astype(str),
        "gender": d["demo_gender"].astype(str),
        "education": d["demo_edu"].astype(str),
        "colorblind": d["demo_colorblind"].astype(str),
        "art_interest": pd.to_numeric(d["Art Interest VAIAK"], errors="coerce"),
    })
    # demographics are constant per participant -> keep the first
    return out.drop_duplicates("userId").set_index("userId")


_LOADERS = {"PARA": _para, "EVA": _eva, "LAPIS": _lapis}


def load_attributes(dataset: str) -> pd.DataFrame:
    return _LOADERS[dataset]()
