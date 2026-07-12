"""Pipeline: load personas from a CSV, show them an image, and log how much each one likes it.

Run from the repo root:

    python src/pipeline.py [image] [--personas-csv data/personas.csv]
                            [--model-name Qwen/Qwen2-VL-7B-Instruct] [--limit N]

``image`` defaults to data/images/img_001.jpg. Each row of the personas CSV
(columns: person_id, description) becomes a Person sharing one Qwen2-VL
backend. Every persona's score (0-100) and comment are printed as they come
in and are also written to a JSON log under data/logs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from persona import Person, QwenVLBackend

DEFAULT_IMAGE = "data/images/img_001.jpg"
DEFAULT_PERSONAS_CSV = "data/personas.csv"
DEFAULT_LOG_DIR = Path("data/logs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show an image to a set of CSV-defined personas and log how much each one likes it."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Path to the image to show every persona (default: %(default)s).",
    )
    parser.add_argument(
        "--personas-csv",
        default=DEFAULT_PERSONAS_CSV,
        help="CSV file with 'person_id' and 'description' columns (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Qwen2-VL model name to load (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N personas from the CSV (useful for quick testing).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON log. Defaults to data/logs/<image_stem>_<timestamp>.json.",
    )
    return parser.parse_args()


def load_personas(csv_path: Path) -> list[dict]:
    """Read persona rows (person_id, description) from a CSV file."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"person_id", "description"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must have {sorted(required)} columns, found {reader.fieldnames}"
            )
        return [
            {"person_id": row["person_id"].strip(), "description": row["description"].strip()}
            for row in reader
            if row["description"].strip()
        ]


_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+)')
_COMMENT_RE = re.compile(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_rating(raw_response: str) -> tuple[Optional[int], str]:
    """Parse a persona's raw response into ``(score, comment)``.

    The persona is instructed to reply with a single JSON object, but small
    VLMs sometimes wrap it in markdown fences or add stray text around it, so
    this tries a strict ``json.loads`` on the extracted ``{...}`` span first
    and falls back to regexes over the raw text if that fails.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()

    start, end = text.find("{"), text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end > start else text

    try:
        data = json.loads(candidate)
        score = max(0, min(100, int(data["score"])))
        comment = str(data["comment"]).strip()
        return score, comment
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    score_match = _SCORE_RE.search(text)
    comment_match = _COMMENT_RE.search(text)
    score = max(0, min(100, int(score_match.group(1)))) if score_match else None
    comment = comment_match.group(1) if comment_match else text
    return score, comment


def _default_output_path(image_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LOG_DIR / f"{image_path.stem}_{timestamp}.json"


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    personas = load_personas(Path(args.personas_csv))
    if args.limit is not None:
        personas = personas[: args.limit]
    if not personas:
        raise ValueError(f"No personas loaded from {args.personas_csv}")

    print(f"Loaded {len(personas)} personas from {args.personas_csv}")
    print(f"Loading backend '{args.model_name}'...")
    backend = QwenVLBackend(model_name=args.model_name)

    results = []
    for row in personas:
        person = Person(
            name=f"Person {row['person_id']}",
            description=row["description"],
            backend=backend,
        )
        raw_response = person.get_opinion(image_path, max_new_tokens=200)
        score, comment = parse_rating(raw_response)

        score_display = f"{score}/100" if score is not None else "UNPARSED"
        print(f"\n=== {person.name} ({score_display}) ===\n{comment}")

        results.append(
            {
                "person_id": row["person_id"],
                "name": person.name,
                "description": row["description"],
                "score": score,
                "comment": comment,
                "raw_response": raw_response,
            }
        )

    scored = [r["score"] for r in results if r["score"] is not None]
    if scored:
        print(
            f"\nAverage score: {sum(scored) / len(scored):.1f}/100 "
            f"({len(scored)}/{len(results)} parsed)"
        )

    output_path = Path(args.output) if args.output else _default_output_path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "image": str(image_path),
        "model_name": args.model_name,
        "personas_csv": str(args.personas_csv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nWrote log to {output_path}")


if __name__ == "__main__":
    main()
