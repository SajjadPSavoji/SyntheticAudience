"""Push local dataset(s) to private, HF-native Hugging Face dataset repos.

Usage:
    python scripts/push_to_hf.py all            # every known dataset (one command)
    python scripts/push_to_hf.py lapis          # one dataset
    python scripts/push_to_hf.py lapis eva      # several

Each dataset repo gets a hybrid, HF-native layout:
  * an ``images`` config (parquet with the ORIGINAL image bytes embedded) —
    browsable in the Hub viewer and loadable with `load_dataset`; and
  * every other file (CSV tables, readmes, licenses) stored verbatim, so quirky
    CSVs round-trip byte-for-byte.

`datasets.push_to_hub` shards the images into parquet and commits efficiently, so
this sidesteps the Hub's 10k-files-per-directory and 128-commits/hour limits that
broke the file-by-file approach. A ``layout_manifest.json`` records the layout so
`fetch_from_hf.py` can rebuild data/<name>/ exactly.

Each repo is deleted and recreated for a clean slate. A failure on one dataset
does not stop the rest; the script exits non-zero if any failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Image
from huggingface_hub import HfApi

from hf_dataset import (
    DATASETS, MANIFEST_NAME, RAW_PREFIX, ROOT, DatasetSpec, get_token, resolve, selected,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def _junk(rel: Path) -> bool:
    return any(p == ".DS_Store" or p == ".cache" or p.startswith(".") for p in rel.parts)


def _images(spec: DatasetSpec) -> list[Path]:
    root = spec.local_dir / spec.images_dir
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not _junk(p.relative_to(root))
    )


def _verbatim(spec: DatasetSpec) -> list[Path]:
    """Every file NOT under the images dir (and not junk): CSVs, readmes, etc."""
    images_root = spec.local_dir / spec.images_dir
    out = []
    for p in sorted(spec.local_dir.rglob("*")):
        if not p.is_file() or _junk(p.relative_to(spec.local_dir)):
            continue
        if images_root in p.parents:
            continue
        out.append(p)
    return out


def _tree_files(root: Path) -> list[Path]:
    """Every non-junk file under `root`, sorted."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not _junk(p.relative_to(root))
    )


def _dataset_card(spec: DatasetSpec, counts: list[tuple[str, Path, int, int]]) -> str:
    """Minimal card for verbatim repos (no push_to_hub to generate one for us)."""
    rows = "\n".join(
        f"| `{RAW_PREFIX}/{prefix}/` | `{rel}` | {n:,} | {mb:,.0f} MB |"
        for prefix, rel, n, mb in counts
    )
    return f"""---
license: other
viewer: false
tags:
  - autopolish
  - computational-aesthetics
  - experiment-outputs
---

# {spec.name.upper()}

{spec.description}

This repo holds **experiment outputs**, not a source corpus. Files are stored
verbatim with their directory layout intact, because the analysis scripts in
`scripts/analysis/` read them by path (for example
`c4_trajectory.py --output-root <runs>/c4_run2`).

| Repo prefix | Restores to | Files | Size |
|---|---|---|---|
{rows}

`layout_manifest.json` records that mapping. To rebuild the local trees:

```bash
python scripts/fetch_from_hf.py {spec.name}
```

The edited images are derivatives of the PARA and EVA photographs, so this repo
is **private** and inherits those datasets' terms.
"""


def push_verbatim(api: HfApi, spec: DatasetSpec, reset: bool) -> None:
    """Upload each source tree file-for-file under raw/<prefix>/, layout intact."""
    counts = []
    for prefix, root in spec.roots():
        if not root.exists() or not any(root.iterdir()):
            raise RuntimeError(f"{root} is missing or empty")
        files = _tree_files(root)
        mb = sum(p.stat().st_size for p in files) / 1e6
        counts.append((prefix, str(root.relative_to(ROOT)), len(files), mb))
        print(f"  {prefix}/ <- {root.relative_to(ROOT)}: {len(files):,} files, {mb:,.0f} MB")

    if reset:
        api.delete_repo(repo_id=spec.repo_id, repo_type="dataset", missing_ok=True)
    api.create_repo(repo_id=spec.repo_id, repo_type="dataset", private=True, exist_ok=True)

    manifest_files: dict[str, list[str]] = {}
    for prefix, root in spec.roots():
        files = _tree_files(root)
        manifest_files[prefix] = [str(p.relative_to(root)) for p in files]
        print(f"  uploading {prefix}/ ({len(files):,} files)...")
        api.upload_folder(
            folder_path=str(root), repo_id=spec.repo_id, repo_type="dataset",
            path_in_repo=f"{RAW_PREFIX}/{prefix}",
            ignore_patterns=["**/.*", ".*", "**/__pycache__/**"],
            commit_message=f"Upload {prefix} tree ({len(files)} files)",
        )

    manifest = {
        "name": spec.name,
        "images_dir": None,
        "sources": [[prefix, rel] for prefix, rel in spec.sources],
        "files": manifest_files,
    }
    api.upload_file(
        path_or_fileobj=BytesIO(json.dumps(manifest, indent=2).encode()),
        path_in_repo=MANIFEST_NAME, repo_id=spec.repo_id, repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=BytesIO(_dataset_card(spec, counts).encode()),
        path_in_repo="README.md", repo_id=spec.repo_id, repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{spec.repo_id}")


def push_one(api: HfApi, token: str, spec: DatasetSpec, reset: bool = True) -> None:
    if spec.verbatim_only:
        print(f"{spec.name}: verbatim tree(s) -> {spec.repo_id}")
        push_verbatim(api, spec, reset)
        return

    local = spec.local_dir
    if not local.exists() or not any(local.iterdir()):
        raise RuntimeError(f"{local} is missing or empty")

    images = _images(spec)
    files = _verbatim(spec)
    print(f"{spec.name}: {len(images)} images + {len(files)} file(s) -> {spec.repo_id}")

    # Clean slate so no leftovers from earlier attempts linger.
    api.delete_repo(repo_id=spec.repo_id, repo_type="dataset", missing_ok=True)
    api.create_repo(repo_id=spec.repo_id, repo_type="dataset", private=True, exist_ok=True)

    # 1. images config (original bytes embedded, lossless).
    img_root = local / spec.images_dir
    ds = Dataset.from_dict({
        "file_name": [str(p.relative_to(img_root)) for p in images],
        "image": [str(p) for p in images],
    }).cast_column("image", Image())
    print(f"  pushing images config ({len(images)} images)...")
    ds.push_to_hub(spec.repo_id, config_name="images", private=True, token=token)

    # 2. verbatim files (CSVs, readme, license) under raw/, keeping their tree.
    rels = []
    for p in files:
        rel = str(p.relative_to(local))
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{RAW_PREFIX}/{rel}",
                        repo_id=spec.repo_id, repo_type="dataset")
        rels.append(rel)
    if rels:
        print(f"  uploaded {len(rels)} verbatim file(s)")

    # 3. manifest so fetch can rebuild data/<name>/ exactly.
    manifest = {"name": spec.name, "images_dir": spec.images_dir, "files": rels}
    api.upload_file(
        path_or_fileobj=BytesIO(json.dumps(manifest, indent=2).encode()),
        path_in_repo=MANIFEST_NAME, repo_id=spec.repo_id, repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{spec.repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Push dataset(s) to private HF-native repos.")
    ap.add_argument("datasets", nargs="+",
                    help=f"dataset name(s) or 'all'. Known: {', '.join(DATASETS)}")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing repo instead of deleting it first; "
                         "re-uploads only what changed (verbatim datasets only).")
    args = ap.parse_args()

    names = selected(args.datasets)
    token = get_token()
    api = HfApi(token=token)

    failed: dict[str, str] = {}
    for name in names:
        try:
            push_one(api, token, resolve(name), reset=not args.resume)
        except Exception as exc:  # keep going so one bad dataset doesn't block the rest
            failed[name] = str(exc)
            print(f"FAILED {name}: {exc}", file=sys.stderr)

    ok = [n for n in names if n not in failed]
    print(f"\nSummary: {len(ok)}/{len(names)} pushed"
          f"{' (' + ', '.join(ok) + ')' if ok else ''}.")
    if failed:
        for name, err in failed.items():
            print(f"  - {name}: {err.splitlines()[0]}", file=sys.stderr)
        print("Re-run the same command to retry the failed datasets.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
