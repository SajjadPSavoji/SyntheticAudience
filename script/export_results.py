#!/usr/bin/env python
"""Combine a chunked PARA log into ONE json file with every result inline.

A run writes a small summary (para_<name>.json: config + metrics + a manifest of
chunk files) plus the ratings in para_<name>.part-NNNN.json chunk files. This
script stitches them back into a single self-contained JSON — handy for sharing
or loading the whole thing at once.

    python script/export_results.py data/logs/para_1000.json
    python script/export_results.py data/logs/para_1000.json -o data/logs/para_1000_full.json

By default it writes <summary_stem>_full.json next to the summary. Works on a run
that is still in progress (you just get whatever has been checkpointed so far),
and on old single-file logs (which are already combined — it just copies them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from para_pipeline import _read_log_and_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", help="Path to the run's summary JSON (e.g. data/logs/para_1000.json)")
    ap.add_argument("-o", "--output", default=None, help="Output path (default: <stem>_full.json)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    log, results = _read_log_and_results(summary_path)

    # One self-contained object: all the summary fields, but with results inline
    # and the chunk-manifest keys dropped (they no longer apply to a single file).
    combined = {k: v for k, v in log.items() if k not in ("result_parts", "chunk_size")}
    combined["n_ratings"] = len(results)
    combined["results"] = results

    out = Path(args.output) if args.output else summary_path.with_name(summary_path.stem + "_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} results to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
