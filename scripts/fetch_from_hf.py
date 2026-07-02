"""Fetch dataset(s) from private HF-native repos back into data/<name>/.

Usage:
    python scripts/fetch_from_hf.py all          # every known dataset
    python scripts/fetch_from_hf.py lapis        # one dataset
    python scripts/fetch_from_hf.py lapis eva    # several

Mirror of push_to_hf.py: reads the repo's layout_manifest.json, writes the images
back from the embedded ORIGINAL bytes (lossless) under the images dir, and copies
every verbatim file (CSVs, readmes) to its original relative path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from datasets import Image, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from hf_dataset import DATASETS, MANIFEST_NAME, RAW_PREFIX, DatasetSpec, get_token, resolve, selected


def fetch_one(token: str, spec: DatasetSpec) -> None:
    repo = spec.repo_id
    local = spec.local_dir
    local.mkdir(parents=True, exist_ok=True)

    with open(hf_hub_download(repo, MANIFEST_NAME, repo_type="dataset", token=token)) as f:
        manifest = json.load(f)
    print(f"{repo}: images + {len(manifest['files'])} file(s) -> {local}")

    # images: write original bytes back under images_dir, preserving relative paths.
    img_root = local / manifest["images_dir"]
    ds = load_dataset(repo, "images", split="train", token=token)
    ds = ds.cast_column("image", Image(decode=False))  # keep raw bytes, don't re-encode
    print(f"  restoring {len(ds)} images...")
    for ex in ds:
        out = img_root / ex["file_name"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(ex["image"]["bytes"])

    # verbatim files (CSVs, readme, license) restored from under raw/.
    for rel in manifest["files"]:
        src = hf_hub_download(repo, f"{RAW_PREFIX}/{rel}", repo_type="dataset", token=token)
        out = local / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    if manifest["files"]:
        print(f"  restored {len(manifest['files'])} verbatim file(s)")

    print(f"Done: {local}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch dataset(s) from private HF-native repos.")
    ap.add_argument("datasets", nargs="+",
                    help=f"dataset name(s) or 'all'. Known: {', '.join(DATASETS)}")
    args = ap.parse_args()

    names = selected(args.datasets)
    token = get_token()
    _ = HfApi(token=token)  # validates token early

    failed: dict[str, str] = {}
    for name in names:
        try:
            fetch_one(token, resolve(name))
        except Exception as exc:
            failed[name] = str(exc)
            print(f"FAILED {name}: {exc}", file=sys.stderr)

    ok = [n for n in names if n not in failed]
    print(f"\nSummary: {len(ok)}/{len(names)} fetched"
          f"{' (' + ', '.join(ok) + ')' if ok else ''}.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
