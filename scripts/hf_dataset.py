"""Shared config + dataset registry for the <-> Hugging Face sync scripts.

Every dataset lives locally at ``data/<name>/`` and is published to a private Hub
dataset repo ``<HF_OWNER>/<NAME>`` in a hybrid, HF-native layout:

  * the images become a real ``datasets`` ``images`` config (parquet with the
    original image bytes embedded) — browsable in the Hub viewer and loadable
    with ``load_dataset``; and
  * every other file (CSV tables, readmes, licenses) is stored verbatim, so
    quirky CSVs (BOMs, missing headers, mixed delimiters) round-trip exactly.

This module is the single source of truth so `push` and `fetch` never drift.
To add a dataset, drop it under `data/` and add one line to ``DATASETS`` naming
its images subdirectory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this scripts/ directory; datasets live under data/.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

HF_OWNER = os.getenv("HF_OWNER", "savoji").strip()

# Small JSON manifest written to each repo so `fetch` can rebuild data/<name>/.
MANIFEST_NAME = "layout_manifest.json"

# Verbatim files (CSVs, readmes) are stored under this repo prefix so they never
# collide with the auto-generated dataset card (README.md) or the parquet dirs.
RAW_PREFIX = "raw"


class DatasetSpec:
    """How one local dataset maps to its Hub repo. ``images_dir`` is the only
    per-dataset knob: the folder (relative to data/<name>/) holding the images;
    everything else in the dataset is treated as a verbatim file."""

    def __init__(self, name: str, images_dir: str):
        self.name = name
        self.repo_id = f"{HF_OWNER}/{name.upper()}"
        self.images_dir = images_dir

    @property
    def local_dir(self) -> Path:
        return DATA_DIR / self.name


DATASETS: dict[str, DatasetSpec] = {
    "lapis": DatasetSpec("lapis", images_dir="images"),
    "eva": DatasetSpec("eva", images_dir="images"),
    "para": DatasetSpec("para", images_dir="imgs"),
}


def get_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Copy .env.example to .env and add your "
            "Hugging Face token (https://huggingface.co/settings/tokens)."
        )
    return token


def resolve(name: str) -> DatasetSpec:
    """Return the DatasetSpec for a name, or exit with a clear error."""
    key = name.lower()
    if key not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)}.")
    return DATASETS[key]


def selected(names: list[str]) -> list[str]:
    """Expand CLI args into concrete dataset names; 'all' means every known one."""
    if names == ["all"]:
        return list(DATASETS)
    return [n.lower() for n in names]
